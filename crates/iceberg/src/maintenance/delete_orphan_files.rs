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

//! `DeleteOrphanFiles`: list a table location and delete every file that no valid snapshot
//! references. Rust port of Java `DeleteOrphanFilesSparkAction`, without the Spark distribution
//! layer.
//!
//! **This action deletes files, and nothing rolls the deletion back.** A reachability omission, a
//! URI-normalization mistake, or an over-broad listing deletes live table data. Every eligibility
//! decision here biases toward under-deletion.
//!
//! Which facts rest on 1.10.0 BYTECODE and which on tagless MAIN source:
//! [`task/delete-orphan-files-java-provenance.md`]. The action class itself has no 1.10.0 Spark
//! bytecode to pin against; its load-bearing helpers do.
//!
//! # The algorithm (Java `doExecute`)
//!
//! | step | rule |
//! |---|---|
//! | valid-file universe | every content file, manifest, and manifest list of every snapshot, the current metadata.json, the metadata-log previous-file entries, the version hint, and every statistics file |
//! | list | list the location and drop hidden paths (Java `PartitionAwareHiddenPathFilter`) |
//! | age cut | keep only `created_at_millis < older_than`, which protects a write in flight |
//! | join | normalize both sides to a `FileUri`, join on path, classify by [`PrefixMismatchMode`] |
//! | delete | delete each orphan and collect per-file failures |
//!
//! The universe takes **every manifest entry, including `DELETED` tombstones**. Java's
//! `contentFileDS` never calls `liveEntries()`. A tombstone keeps a file out of the orphan set.
//! This is the load-bearing difference from `expire_cleanup`, which subtracts on `is_alive()`.
//! Never filter this universe on liveness. The metadata-log walk is not recursive, matching
//! Java's `otherMetadataFileDS` default.
//!
//! # Defaults (Java parity)
//!
//! | setting | default |
//! |---|---|
//! | `older_than` | `now − 3 days` |
//! | `prefix_mismatch_mode` | [`PrefixMismatchMode::Error`] |
//! | `equal_schemes` | `{s3n → s3, s3a → s3}`, merged with user entries; a user key wins |
//! | `equal_authorities` | empty; user entries replace it wholesale |
//! | `location` | the table's location |
//! | delete | [`FileIO::delete`](crate::io::FileIO::delete) |
//!
//! Java's constructor throws `ValidationException` when `gc.enabled=false`. This port refuses at
//! [`Self::execute`] with the same message.
//!
//! # Failure posture
//!
//! Java's `deleteNonBulk` suppresses a per-file delete failure and continues, and it collects the
//! orphan list before deletion. This port collects failures in
//! [`DeleteOrphanFilesResult::delete_failures`] instead of logging them. The crate has no logging
//! facade, and a deletion sweep must not swallow an error. A planning-stage failure returns `Err`
//! before any deletion.
//!
//! # Deferred
//!
//! The sweep is sequential: Java's `executeDeleteWith(ExecutorService)` is throughput, not
//! correctness. The fork's [`FileIO`] has no bulk-delete surface. `compareToFileList` and
//! streaming results are not ported.

use std::collections::{HashMap, HashSet};

use futures::future::BoxFuture;

use crate::error::Result;
use crate::io::FileIO;
use crate::spec::{PartitionSpecRef, TableProperties};
use crate::table::Table;
use crate::{Error, ErrorKind};

/// The default `older_than` grace period (Java `TimeUnit.DAYS.toMillis(3)`).
const DEFAULT_OLDER_THAN_AGE_MILLIS: i64 = 3 * 24 * 60 * 60 * 1000;

/// The injected delete function: receives a file location, resolves to a deletion outcome. The
/// default deletes through [`FileIO::delete`].
pub type OrphanDeleteFunction = dyn Fn(String) -> BoxFuture<'static, Result<()>> + Send + Sync;

/// How [`DeleteOrphanFiles`] treats a listed file whose path matches a valid file but whose
/// scheme or authority differs after normalization. Java `DeleteOrphanFiles.PrefixMismatchMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixMismatchMode {
    /// Fail the action when any prefix conflict remains (Java `ERROR`). This is the default. The
    /// error names each conflicting pair, so the operator can resolve it through
    /// [`DeleteOrphanFiles::equal_schemes`] or [`DeleteOrphanFiles::equal_authorities`].
    Error,
    /// Treat a prefix-conflicting file as NOT orphan and skip it (Java `IGNORE`).
    Ignore,
    /// Treat a prefix-conflicting file as orphan and delete it (Java `DELETE`). The deletion is
    /// unrecoverable.
    Delete,
}

impl PrefixMismatchMode {
    /// Parses a mode case-insensitively (Java `PrefixMismatchMode.fromString`). An unknown value
    /// gives a `DataInvalid` error with Java's message.
    pub fn from_string(mode: &str) -> Result<Self> {
        match mode.to_uppercase().as_str() {
            "ERROR" => Ok(PrefixMismatchMode::Error),
            "IGNORE" => Ok(PrefixMismatchMode::Ignore),
            "DELETE" => Ok(PrefixMismatchMode::Delete),
            _ => Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Invalid mode: {mode}"),
            )),
        }
    }
}

/// The outcome of a [`DeleteOrphanFiles::execute`] sweep (Java `Result.orphanFileLocations()`),
/// plus every collected per-file delete failure.
///
/// `orphan_file_locations` holds the full orphan set the join produced. Java collects that list
/// before deletion, so a path appears here even when its own deletion failed.
#[derive(Debug, Default)]
pub struct DeleteOrphanFilesResult {
    /// Locations of all orphan files (the listed-but-unreferenced set), sorted deterministically.
    pub orphan_file_locations: Vec<String>,
    /// Per-file delete failures. Empty means every orphan deleted cleanly.
    pub delete_failures: Vec<OrphanDeleteFailure>,
}

/// One collected, non-aborting delete failure. Java logs and continues instead.
#[derive(Debug)]
pub struct OrphanDeleteFailure {
    /// The orphan file whose deletion failed.
    pub path: String,
    /// The underlying error.
    pub error: Error,
}

/// A normalized file identity for the orphan join (Java `org.apache.iceberg.actions.FileURI`).
///
/// The join matches on `path` alone. Scheme and authority feed [`PrefixMismatchMode`].
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileUri {
    /// Scheme after `equal_schemes` mapping. `None` for a scheme-less local path.
    scheme: Option<String>,
    /// Authority after `equal_authorities` mapping.
    authority: Option<String>,
    /// The URI path component, and the orphan join key.
    path: String,
    /// The original location string, which is what gets deleted and returned.
    uri_as_string: String,
}

impl FileUri {
    /// Parses `location` into a [`FileUri`], mirroring Java `ToFileURI.toFileURI`. Parsing follows
    /// Hadoop `Path(s).toUri()` over the location shapes Iceberg writers produce.
    ///
    /// | form | scheme | authority | path |
    /// |---|---|---|---|
    /// | `s3://bucket/key` | `s3` | `bucket`, possibly empty | `/key` |
    /// | `file:/tmp/a` | `file` | none | `/tmp/a` |
    /// | `/tmp/a` | none | none | `/tmp/a` |
    ///
    /// The third row is the local-path case. A scheme-less valid path matches an actual path of
    /// any scheme, so a bare stored path and a `file://` listed path never conflict.
    fn parse(
        location: &str,
        equal_schemes: &HashMap<String, String>,
        equal_authorities: &HashMap<String, String>,
    ) -> Self {
        let (raw_scheme, authority, path) = split_uri(location);
        let scheme = raw_scheme.map(|scheme| equal_schemes.get(&scheme).cloned().unwrap_or(scheme));
        let authority = authority.map(|authority| {
            equal_authorities
                .get(&authority)
                .cloned()
                .unwrap_or(authority)
        });
        FileUri {
            scheme,
            authority,
            path,
            uri_as_string: location.to_string(),
        }
    }

    /// Whether this (valid) URI's scheme matches `actual`'s (Java `FileURI.schemeMatch` →
    /// `uriComponentMatch`). A `None`/empty valid scheme matches any actual scheme.
    fn scheme_matches(&self, actual: &FileUri) -> bool {
        uri_component_match(self.scheme.as_deref(), actual.scheme.as_deref())
    }

    /// Whether this (valid) URI's authority matches `actual`'s (Java `FileURI.authorityMatch`).
    fn authority_matches(&self, actual: &FileUri) -> bool {
        uri_component_match(self.authority.as_deref(), actual.authority.as_deref())
    }
}

/// Java `FileURI.uriComponentMatch(valid, actual)`. An absent or empty valid component matches
/// any actual component. Otherwise the two must match case-insensitively.
fn uri_component_match(valid: Option<&str>, actual: Option<&str>) -> bool {
    match valid {
        None | Some("") => true,
        Some(valid) => actual.is_some_and(|actual| valid.eq_ignore_ascii_case(actual)),
    }
}

/// Splits a location into `(scheme, authority, path)`. [`FileUri::parse`] lists the cases.
fn split_uri(location: &str) -> (Option<String>, Option<String>, String) {
    // Hadoop's Path treats a Windows drive letter specially. Iceberg locations are POSIX or
    // object-store URIs, so the RFC-3986 scheme rule is faithful here.
    let scheme_end = scheme_delimiter(location);
    let (scheme, rest) = match scheme_end {
        Some(index) => (Some(location[..index].to_string()), &location[index + 1..]),
        None => (None, location),
    };

    // An authority is present only if the remainder starts with "//".
    if let Some(after_slashes) = rest.strip_prefix("//") {
        // The authority runs to the next '/', '?' or '#'. "file:///tmp/a" gives Some("").
        let authority_end = after_slashes
            .find(['/', '?', '#'])
            .unwrap_or(after_slashes.len());
        let authority = after_slashes[..authority_end].to_string();
        let path = after_slashes[authority_end..].to_string();
        (scheme, Some(authority), path)
    } else {
        // Strip a query or fragment to mirror URI.getPath(). A file location never carries one.
        let path_end = rest.find(['?', '#']).unwrap_or(rest.len());
        (scheme, None, rest[..path_end].to_string())
    }
}

/// Index of the ':' that ends a URI scheme, or `None` when `location` has no scheme.
fn scheme_delimiter(location: &str) -> Option<usize> {
    let bytes = location.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_alphabetic() {
        return None;
    }
    for (index, &byte) in bytes.iter().enumerate() {
        match byte {
            b':' => return if index == 0 { None } else { Some(index) },
            b if b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.') => continue,
            _ => return None,
        }
    }
    None
}

/// The hidden-path filter (Java `FileSystemWalker$PartitionAwareHiddenPathFilter`). A segment is
/// hidden when it starts with `_` or `.`. A partition directory `<field>=...` is exempt when
/// `<field>` names a partition field of any spec whose own name starts with `_` or `.`.
struct PartitionAwareHiddenPathFilter {
    /// `<field>=` prefixes the plain hidden rule would otherwise hide. Empty means that rule
    /// applies unchanged.
    hidden_partition_prefixes: Vec<String>,
}

impl PartitionAwareHiddenPathFilter {
    /// Build the filter from the table's partition specs (Java `forSpecs`).
    fn for_specs<'a>(specs: impl Iterator<Item = &'a PartitionSpecRef>) -> Self {
        let mut hidden_partition_prefixes: Vec<String> = specs
            .flat_map(|spec| spec.fields().iter())
            .filter(|field| field.name.starts_with('_') || field.name.starts_with('.'))
            .map(|field| format!("{}=", field.name))
            .collect();
        hidden_partition_prefixes.sort();
        hidden_partition_prefixes.dedup();
        PartitionAwareHiddenPathFilter {
            hidden_partition_prefixes,
        }
    }

    /// Whether one path segment is visible. Java `accept`.
    fn accepts_segment(&self, segment: &str) -> bool {
        self.is_partition_segment(segment) || !is_plain_hidden(segment)
    }

    /// Whether `segment` is a partition directory exempt from the hidden rule (Java
    /// `isHiddenPartitionPath`).
    fn is_partition_segment(&self, segment: &str) -> bool {
        self.hidden_partition_prefixes
            .iter()
            .any(|prefix| segment.starts_with(prefix.as_str()))
    }

    /// Whether `location`'s path under `base` holds a hidden segment (Java
    /// `FileSystemWalker.isHiddenPath`). Only segments strictly under `base` are checked, so a
    /// hidden component of the table root never disqualifies the whole listing.
    fn is_hidden_under(&self, base: &str, location: &str) -> bool {
        let Some(relative) = relative_under(base, location) else {
            // A listing of `base` should never yield this. Treat it as hidden, so it survives.
            return true;
        };
        // Java's PathFilter applies to every segment from the file up to, but excluding, baseDir.
        // A file named "_x" under the root is therefore hidden too.
        relative
            .split('/')
            .filter(|segment| !segment.is_empty())
            .any(|segment| !self.accepts_segment(segment))
    }
}

/// Java `HiddenPathFilter.accept`: a name is hidden only if it starts with `_` or `.`.
fn is_plain_hidden(segment: &str) -> bool {
    segment.starts_with('_') || segment.starts_with('.')
}

/// The path of `location` strictly under `base`, or `None` when it is not under `base`. A
/// trailing `/` on `base` is tolerated.
fn relative_under<'a>(base: &str, location: &'a str) -> Option<&'a str> {
    let base = base.strip_suffix('/').unwrap_or(base);
    let remainder = location.strip_prefix(base)?;
    // Require a directory boundary, so base "ab" does not match location "ab2/x".
    remainder.strip_prefix('/')
}

/// Deletes orphan metadata, data, and delete files by listing a location and comparing it against
/// the valid-file universe of all snapshots. The module docs carry the algorithm and the defaults.
///
/// **This action deletes files.** Build it with [`DeleteOrphanFiles::new`], configure it, then run
/// [`Self::execute`].
pub struct DeleteOrphanFiles {
    table: Table,
    location: String,
    older_than_millis: i64,
    prefix_mismatch_mode: PrefixMismatchMode,
    equal_schemes: HashMap<String, String>,
    equal_authorities: HashMap<String, String>,
    delete_function: Option<Box<OrphanDeleteFunction>>,
}

impl DeleteOrphanFiles {
    /// Creates the action for `table` with Java's defaults. The module docs list them.
    pub fn new(table: Table) -> Self {
        let location = table.metadata().location().to_string();
        DeleteOrphanFiles {
            table,
            location,
            older_than_millis: now_millis().saturating_sub(DEFAULT_OLDER_THAN_AGE_MILLIS),
            prefix_mismatch_mode: PrefixMismatchMode::Error,
            equal_schemes: default_equal_schemes(),
            equal_authorities: HashMap::new(),
            delete_function: None,
        }
    }

    /// The location to scan (Java `location(String)`). Point it at a subdirectory to sweep only
    /// that subtree.
    pub fn location(mut self, location: impl Into<String>) -> Self {
        self.location = location.into();
        self
    }

    /// Only files older than this epoch-millis timestamp are eligible (Java `olderThan(long)`).
    /// The grace protects a file an in-flight commit adds before it references it.
    pub fn older_than(mut self, older_than_millis: i64) -> Self {
        self.older_than_millis = older_than_millis;
        self
    }

    /// Sets how a prefix conflict is handled (Java `prefixMismatchMode`).
    pub fn prefix_mismatch_mode(mut self, mode: PrefixMismatchMode) -> Self {
        self.prefix_mismatch_mode = mode;
        self
    }

    /// Adds schemes to treat as equal (Java `equalSchemes`). The map merges on top of the
    /// defaults, and a user mapping wins a key collision. Comma-separated keys are flattened.
    pub fn equal_schemes(mut self, equal_schemes: HashMap<String, String>) -> Self {
        let mut merged = default_equal_schemes();
        merged.extend(flatten_map(equal_schemes));
        self.equal_schemes = merged;
        self
    }

    /// Sets authorities to treat as equal (Java `equalAuthorities`). The map replaces the current
    /// one, as Java does. There is no built-in default.
    pub fn equal_authorities(mut self, equal_authorities: HashMap<String, String>) -> Self {
        self.equal_authorities = flatten_map(equal_authorities);
        self
    }

    /// Replaces the delete function (Java `deleteWith(Consumer<String>)`). The function receives
    /// exactly the orphan set, so a caller can collect orphans instead of deleting them.
    pub fn delete_with(
        mut self,
        delete_function: impl Fn(String) -> BoxFuture<'static, Result<()>> + Send + Sync + 'static,
    ) -> Self {
        self.delete_function = Some(Box::new(delete_function));
        self
    }

    /// Plans the orphan set and deletes it. The module docs carry the algorithm and the failure
    /// posture.
    ///
    /// # Errors
    ///
    /// Fails without deleting anything when the `gc.enabled` gate refuses, when planning cannot
    /// read a manifest list or manifest, or when an `ERROR`-mode prefix conflict remains.
    pub async fn execute(self) -> Result<DeleteOrphanFilesResult> {
        // Refuse before any listing or deletion.
        self.check_gc_enabled()?;

        let file_io = self.table.file_io().clone();

        let valid_locations = self.collect_valid_files().await?;

        let listed = self.list_candidate_files(&file_io).await?;

        let orphan_locations = self.find_orphan_files(listed, &valid_locations)?;

        let mut result = DeleteOrphanFilesResult {
            orphan_file_locations: orphan_locations.clone(),
            delete_failures: Vec::new(),
        };
        for path in orphan_locations {
            let outcome = match &self.delete_function {
                Some(delete) => delete(path.clone()).await,
                None => file_io.delete(&path).await,
            };
            if let Err(error) = outcome {
                result
                    .delete_failures
                    .push(OrphanDeleteFailure { path, error });
            }
        }
        Ok(result)
    }

    /// Java's `gc.enabled` gate (constructor `ValidationException.check(... GC_ENABLED ...)`).
    fn check_gc_enabled(&self) -> Result<()> {
        let gc_enabled = parse_bool_property(
            self.table.metadata().properties(),
            TableProperties::PROPERTY_GC_ENABLED,
            TableProperties::PROPERTY_GC_ENABLED_DEFAULT,
        )?;
        if !gc_enabled {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Cannot delete orphan files: GC is disabled (deleting files may corrupt other \
                 tables)"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Builds the valid-file universe, including `DELETED` tombstone entries. The module docs
    /// step 1 lists what it holds.
    ///
    /// A manifest-list or manifest read failure returns `Err` before any deletion can run.
    async fn collect_valid_files(&self) -> Result<HashSet<String>> {
        let metadata = self.table.metadata();
        let file_io = self.table.file_io();
        let mut valid: HashSet<String> = HashSet::new();

        for snapshot in metadata.snapshots() {
            valid.insert(snapshot.manifest_list().to_string());

            let manifest_list = snapshot
                .load_manifest_list(file_io, metadata)
                .await
                .map_err(|error| {
                    error.with_context(
                        "snapshot_id",
                        format!(
                            "failed to read manifest list of snapshot {} while planning \
                             delete-orphan-files (no files were deleted)",
                            snapshot.snapshot_id()
                        ),
                    )
                })?;

            for manifest_file in manifest_list.entries() {
                valid.insert(manifest_file.manifest_path.clone());

                // Take every entry, including a DELETED tombstone. Java reads through
                // ManifestFiles.read, not liveEntries. A tombstoned file is still referenced.
                let manifest = manifest_file
                    .load_manifest(file_io)
                    .await
                    .map_err(|error| {
                        error.with_context(
                            "manifest_path",
                            format!(
                                "failed to read manifest {} while planning delete-orphan-files \
                                 (no files were deleted)",
                                manifest_file.manifest_path
                            ),
                        )
                    })?;
                for entry in manifest.entries() {
                    valid.insert(entry.file_path().to_string());
                }
            }
        }

        // The metadata-log walk is not recursive, matching Java's otherMetadataFileDS default.
        for log_entry in metadata.metadata_log() {
            valid.insert(log_entry.metadata_file.clone());
        }
        if let Some(current_metadata_location) = self.table.metadata_location() {
            valid.insert(current_metadata_location.to_string());
        }
        valid.insert(version_hint_location(metadata.location()));
        for statistics in metadata.statistics_iter() {
            valid.insert(statistics.statistics_path.clone());
        }
        for statistics in metadata.partition_statistics_iter() {
            valid.insert(statistics.statistics_path.clone());
        }

        Ok(valid)
    }

    /// Lists every file under the location that passes the hidden-path filter and the
    /// `older_than` cut.
    async fn list_candidate_files(&self, file_io: &FileIO) -> Result<Vec<crate::io::FileInfo>> {
        let hidden_filter =
            PartitionAwareHiddenPathFilter::for_specs(self.table.metadata().partition_specs_iter());

        // FileIO::list fails loudly when a backend cannot enumerate, and never returns an empty
        // Ok. An orphan decision must never run against a silently empty listing.
        let listed = file_io.list(&self.location).await?;

        Ok(listed
            .into_iter()
            .filter(|file| !hidden_filter.is_hidden_under(&self.location, &file.location))
            .filter(|file| file.created_at_millis < self.older_than_millis)
            .collect())
    }

    /// The orphan join (Java `findOrphanFiles`). Returns sorted orphan locations, or `Err` when
    /// an `ERROR`-mode prefix conflict remains.
    fn find_orphan_files(
        &self,
        listed: Vec<crate::io::FileInfo>,
        valid_locations: &HashSet<String>,
    ) -> Result<Vec<String>> {
        // Several valid locations can share one path under different schemes. Keep all of them,
        // so a listed file matches if any one is prefix-compatible.
        let mut valid_by_path: HashMap<String, Vec<FileUri>> = HashMap::new();
        for location in valid_locations {
            let valid_uri = FileUri::parse(location, &self.equal_schemes, &self.equal_authorities);
            valid_by_path
                .entry(valid_uri.path.clone())
                .or_default()
                .push(valid_uri);
        }

        let mut orphans: Vec<String> = Vec::new();
        // Conflicting (valid, actual) pairs, for the ERROR-mode message.
        let mut scheme_conflicts: HashSet<(String, String)> = HashSet::new();
        let mut authority_conflicts: HashSet<(String, String)> = HashSet::new();

        for file in listed {
            let actual =
                FileUri::parse(&file.location, &self.equal_schemes, &self.equal_authorities);
            match valid_by_path.get(&actual.path) {
                None => orphans.push(actual.uri_as_string.clone()),
                Some(valid_candidates) => {
                    let classification = classify_against_valid(
                        &actual,
                        valid_candidates,
                        self.prefix_mismatch_mode,
                        &mut scheme_conflicts,
                        &mut authority_conflicts,
                    );
                    if let OrphanClassification::Orphan = classification {
                        orphans.push(actual.uri_as_string.clone());
                    }
                }
            }
        }

        // Any remaining conflict fails the whole action, as Java's ValidationException does.
        if self.prefix_mismatch_mode == PrefixMismatchMode::Error
            && (!scheme_conflicts.is_empty() || !authority_conflicts.is_empty())
        {
            return Err(prefix_conflict_error(
                &scheme_conflicts,
                &authority_conflicts,
            ));
        }

        orphans.sort();
        orphans.dedup();
        Ok(orphans)
    }
}

/// Whether a path-matched listed file is orphan.
enum OrphanClassification {
    Orphan,
    NotOrphan,
}

/// Classifies a path-matched `actual` against its candidates (Java
/// `FindOrphanFiles.toOrphanFile`).
///
/// Java's left-outer join pairs one actual file with at most one valid file per path. The Rust
/// universe is a set, so a path can carry several valid locations. A listed file is not orphan
/// when any candidate is prefix-compatible. That bias is deliberate: it under-deletes.
fn classify_against_valid(
    actual: &FileUri,
    valid_candidates: &[FileUri],
    mode: PrefixMismatchMode,
    scheme_conflicts: &mut HashSet<(String, String)>,
    authority_conflicts: &mut HashSet<(String, String)>,
) -> OrphanClassification {
    let mut any_scheme_conflict: Option<(String, String)> = None;
    let mut any_authority_conflict: Option<(String, String)> = None;

    for valid in valid_candidates {
        let scheme_match = valid.scheme_matches(actual);
        let authority_match = valid.authority_matches(actual);
        if scheme_match && authority_match {
            return OrphanClassification::NotOrphan;
        }
        // Keep one representative conflict, in case no candidate fully matches.
        if !scheme_match && any_scheme_conflict.is_none() {
            any_scheme_conflict = Some((
                valid.scheme.clone().unwrap_or_default(),
                actual.scheme.clone().unwrap_or_default(),
            ));
        }
        if !authority_match && any_authority_conflict.is_none() {
            any_authority_conflict = Some((
                valid.authority.clone().unwrap_or_default(),
                actual.authority.clone().unwrap_or_default(),
            ));
        }
    }

    // No candidate fully matched, so this path has a prefix conflict.
    match mode {
        PrefixMismatchMode::Delete => OrphanClassification::Orphan,
        // Record the conflict and delete nothing now. ERROR raises it after the join; IGNORE
        // leaves the file alone.
        PrefixMismatchMode::Error | PrefixMismatchMode::Ignore => {
            if let Some(conflict) = any_scheme_conflict {
                scheme_conflicts.insert(conflict);
            }
            if let Some(conflict) = any_authority_conflict {
                authority_conflicts.insert(conflict);
            }
            OrphanClassification::NotOrphan
        }
    }
}

/// The ERROR-mode prefix-conflict error. The message is Java's text plus the conflicting pairs.
fn prefix_conflict_error(
    scheme_conflicts: &HashSet<(String, String)>,
    authority_conflicts: &HashSet<(String, String)>,
) -> Error {
    let mut conflicts: Vec<String> = scheme_conflicts
        .iter()
        .chain(authority_conflicts.iter())
        .map(|(valid, actual)| format!("({valid}, {actual})"))
        .collect();
    conflicts.sort();
    Error::new(
        ErrorKind::DataInvalid,
        format!(
            "Unable to determine whether certain files are orphan. Metadata references files that \
             match listed/provided files except for authority/scheme. Please, inspect the \
             conflicting authorities/schemes and provide which of them are equal by further \
             configuring the action via equalSchemes() and equalAuthorities() methods. Set the \
             prefix mismatch mode to 'IGNORE' to skip remaining locations with conflicting \
             authorities/schemes or to 'DELETE' iff you are ABSOLUTELY confident that remaining \
             conflicting authorities/schemes are different. It will be impossible to recover \
             deleted files. Conflicting authorities/schemes: [{}].",
            conflicts.join(", ")
        ),
    )
}

/// Java `EQUAL_SCHEMES_DEFAULT = ImmutableMap.of("s3n,s3a", "s3")`, comma-flattened to
/// `{s3n → s3, s3a → s3}`.
fn default_equal_schemes() -> HashMap<String, String> {
    HashMap::from([
        ("s3n".to_string(), "s3".to_string()),
        ("s3a".to_string(), "s3".to_string()),
    ])
}

/// Flattens comma-separated keys (Java `flattenMap`), trimming each split key and the value.
fn flatten_map(map: HashMap<String, String>) -> HashMap<String, String> {
    let mut flattened = HashMap::new();
    for (key, value) in map {
        let value = value.trim().to_string();
        for split_key in key.split(',') {
            flattened.insert(split_key.trim().to_string(), value.clone());
        }
    }
    flattened
}

/// The version-hint location (Java `ReachableFileUtil.versionHintLocation`). Only a Hadoop table
/// has one, but Java always adds it, so a stray hint file is never deleted.
fn version_hint_location(table_location: &str) -> String {
    let trimmed = table_location.strip_suffix('/').unwrap_or(table_location);
    format!("{trimmed}/metadata/version-hint.text")
}

/// Parses a boolean table property (Java `PropertyUtil.propertyAsBoolean`). An unparsable value
/// is a loud error, never a silent default: a typo must not bypass the GC gate.
fn parse_bool_property(
    properties: &HashMap<String, String>,
    key: &str,
    default: bool,
) -> Result<bool> {
    match properties.get(key) {
        None => Ok(default),
        Some(value) => value.parse::<bool>().map_err(|error| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Invalid boolean value '{value}' for table property '{key}'"),
            )
            .with_source(error)
        }),
    }
}

/// Wall-clock epoch millis, saturating on pre-epoch and overflow.
fn now_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => i64::try_from(duration.as_millis()).unwrap_or(i64::MAX),
        Err(_) => 0,
    }
}

/// Test accessors for the private helpers, so a test can pin the corruption-class logic without a
/// full table fixture.
#[cfg(test)]
pub(super) mod test_hooks {
    use std::collections::HashMap;

    use super::{
        FileUri, OrphanClassification, PartitionAwareHiddenPathFilter, PrefixMismatchMode,
        classify_against_valid, default_equal_schemes, flatten_map, split_uri,
        version_hint_location,
    };

    /// A thin test handle over the private [`FileUri`].
    #[derive(Debug, Clone)]
    pub struct FileUriProbe(FileUri);

    impl FileUriProbe {
        /// Parse a location into a normalized URI (see [`FileUri::parse`]).
        pub fn parse(
            location: &str,
            equal_schemes: &HashMap<String, String>,
            equal_authorities: &HashMap<String, String>,
        ) -> Self {
            FileUriProbe(FileUri::parse(location, equal_schemes, equal_authorities))
        }

        /// Whether this (valid) URI's scheme matches `actual`'s.
        pub fn scheme_matches(&self, actual: &FileUriProbe) -> bool {
            self.0.scheme_matches(&actual.0)
        }

        /// Whether this (valid) URI's authority matches `actual`'s.
        pub fn authority_matches(&self, actual: &FileUriProbe) -> bool {
            self.0.authority_matches(&actual.0)
        }

        /// The normalized path component (the join key).
        pub fn path_probe(&self) -> &str {
            &self.0.path
        }
    }

    /// Split a location into `(scheme, authority, path)`.
    pub fn split_uri_probe(location: &str) -> (Option<String>, Option<String>, String) {
        split_uri(location)
    }

    /// The default equal-schemes map.
    pub fn default_equal_schemes_probe() -> HashMap<String, String> {
        default_equal_schemes()
    }

    /// Comma-flatten a map.
    pub fn flatten_map_probe(map: HashMap<String, String>) -> HashMap<String, String> {
        flatten_map(map)
    }

    /// The version-hint location for a table location.
    pub fn version_hint_probe(table_location: &str) -> String {
        version_hint_location(table_location)
    }

    /// Whether `location`'s path under `base` is hidden, given the named-partition exception
    /// prefixes (e.g. `["_part="]`).
    pub fn is_hidden_under_probe(
        base: &str,
        location: &str,
        partition_prefixes: &[String],
    ) -> bool {
        let filter = PartitionAwareHiddenPathFilter {
            hidden_partition_prefixes: partition_prefixes.to_vec(),
        };
        filter.is_hidden_under(base, location)
    }

    /// `(valid, actual)` conflict pairs for a single component (scheme or authority).
    type ConflictPairs = Vec<(String, String)>;

    /// The result of [`classify_one`]: `(is_orphan, scheme_conflicts, authority_conflicts)`.
    type ClassifyResult = (bool, ConflictPairs, ConflictPairs);

    /// Classify `actual` against `valid_candidates` under `mode`, returning
    /// `(is_orphan, scheme_conflicts, authority_conflicts)` (each conflict is a `(valid, actual)`
    /// pair).
    pub fn classify_one(
        actual: &FileUriProbe,
        valid_candidates: &[FileUriProbe],
        mode: PrefixMismatchMode,
    ) -> ClassifyResult {
        let valids: Vec<FileUri> = valid_candidates
            .iter()
            .map(|probe| probe.0.clone())
            .collect();
        let mut scheme_conflicts = std::collections::HashSet::new();
        let mut authority_conflicts = std::collections::HashSet::new();
        let classification = classify_against_valid(
            &actual.0,
            &valids,
            mode,
            &mut scheme_conflicts,
            &mut authority_conflicts,
        );
        let is_orphan = matches!(classification, OrphanClassification::Orphan);
        let mut schemes: Vec<(String, String)> = scheme_conflicts.into_iter().collect();
        let mut authorities: Vec<(String, String)> = authority_conflicts.into_iter().collect();
        schemes.sort();
        authorities.sort();
        (is_orphan, schemes, authorities)
    }
}
