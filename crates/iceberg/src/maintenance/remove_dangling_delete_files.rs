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

//! The [`RemoveDanglingDeleteFiles`] maintenance action. It ports Java's
//! `RemoveDanglingDeletesSparkAction`.
//!
//! A delete file is dangling when its deletes can no longer apply to any live data file. Removing it
//! is garbage collection, and the read result does not change. The inverse is corruption: removing a
//! delete file that still applies resurrects the rows it masks. Every test below pins both
//! directions.
//!
//! # The dangling predicate
//!
//! Group the current snapshot's live DATA entries by `(spec_id, partition)`. Take the minimum data
//! sequence number per group. Left-join each live DELETE entry on the same key. The spec id is part
//! of the key, so two files in one partition tuple under different specs do not share a minimum.
//!
//! | Delete kind | Dangles when |
//! |---|---|
//! | any content type | the group holds no live data file (`min IS NULL`) |
//! | position (content 1) | `delete_seq < min_data_seq`, strict |
//! | equality (content 2) | `delete_seq <= min_data_seq`, non-strict |
//! | deletion vector | `referenced_data_file` is not a live data-file path |
//!
//! The off-by-one is the corruption edge. It complements the read path in
//! [`crate::delete_file_index`]. A position delete applies when `delete_seq >= data_seq`. An equality
//! delete applies only when `delete_seq > data_seq`. Flipping either comparison resurrects same-sequence
//! rows or strands a dangling delete.
//!
//! # Commit vehicle
//!
//! Java runs `table.newRewrite()` and `deleteFile(...)` per dangling file, and records the operation as
//! `Replace`. This action drives the same vehicle through
//! [`RewriteFilesAction`](crate::transaction::rewrite_files). It commits only when the dangling set is
//! non-empty. A plain `RewriteFilesAction` carries every delete manifest forward unchanged, so it never
//! prunes a now-dangling parquet position delete. This action is the dedicated cleaner.
//!
//! Java returns early on a table with one spec that is unpartitioned, because `ManifestFilterManager`
//! already drops such deletes on each commit. This action mirrors that.
//!
//! Removal is metadata-only. The commit tombstones the delete entry in the rewritten DELETE manifest
//! and leaves the bytes on disk, so time travel to an older snapshot still applies the delete.
//! `ExpireSnapshots` and `DeleteOrphanFiles` reclaim the bytes.
//!
//! # A known Java-faithful resurrection race
//!
//! The commit runs no concurrent-conflict validation. A delete-file-only `RewriteFiles` has an empty
//! replaced-data set, and both Java's `BaseRewriteFiles.validate` and `RewriteFilesAction::validate`
//! skip the check in that case. A concurrent sequence-preserving compaction can land a data file at a
//! lower sequence number between the plan and the commit. The delete becomes applicable again, and its
//! removal then resurrects the rows. Java has the identical race, so a guard here would diverge. Run
//! this action when sequence-preserving compaction is not in flight.
//!
//! # Global equality deletes under a multi-spec table
//!
//! A global equality delete carries an unpartitioned spec and an empty partition struct. The read path
//! applies it table-wide when `delete_seq > data_seq`. This action keys it by its own
//! `(spec_id, empty-partition)` group. A table whose live data sits under a different spec therefore
//! flags it dangling even though the reader still honors it. Java joins on `spec_id AND partition`
//! too, so the inconsistency comes from Java. It needs an unpartitioned-to-partitioned spec evolution
//! that leaves a global equality delete live.

use std::collections::{HashMap, HashSet};

use crate::Catalog;
use crate::delete_file_index::referenced_data_file_location;
use crate::error::Result;
use crate::spec::{DataContentType, DataFile, DataFileFormat, Struct};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};

/// The outcome of a [`RemoveDanglingDeleteFiles::execute`] run, with per-content-type counts.
///
/// # Notes
///
/// A no-op run returns this empty and commits no snapshot.
#[derive(Debug, Default, Clone)]
pub struct RemoveDanglingDeleteFilesResult {
    /// Every removed dangling delete file, in discovery order.
    pub removed_delete_files: Vec<DataFile>,
}

impl RemoveDanglingDeleteFilesResult {
    /// Number of removed parquet position-delete files, excluding deletion vectors.
    pub fn removed_position_delete_files_count(&self) -> usize {
        self.removed_delete_files
            .iter()
            .filter(|file| {
                file.content_type() == DataContentType::PositionDeletes
                    && file.file_format() != DataFileFormat::Puffin
            })
            .count()
    }

    /// Number of removed equality-delete files.
    pub fn removed_equality_delete_files_count(&self) -> usize {
        self.removed_delete_files
            .iter()
            .filter(|file| file.content_type() == DataContentType::EqualityDeletes)
            .count()
    }

    /// Number of removed deletion vectors, which are Puffin-format position deletes.
    pub fn removed_dvs_count(&self) -> usize {
        self.removed_delete_files
            .iter()
            .filter(|file| {
                file.content_type() == DataContentType::PositionDeletes
                    && file.file_format() == DataFileFormat::Puffin
            })
            .count()
    }
}

/// The `RemoveDanglingDeleteFiles` maintenance action. It removes, in one `Replace` snapshot, every
/// delete file in the current snapshot that can no longer apply to any live data file. The module docs
/// carry the dangling predicate.
pub struct RemoveDanglingDeleteFiles {
    table: Table,
}

impl RemoveDanglingDeleteFiles {
    /// Create a `RemoveDanglingDeleteFiles` action for `table`.
    pub fn new(table: Table) -> Self {
        RemoveDanglingDeleteFiles { table }
    }

    /// Find the dangling delete files in the current snapshot and remove them in one `Replace` snapshot.
    ///
    /// # Notes
    ///
    /// The run commits nothing and returns an empty result when nothing dangles, when the table has no
    /// current snapshot, or when the table has one unpartitioned spec.
    ///
    /// # Errors
    ///
    /// Fails when reading the manifests fails, or when the commit fails.
    pub async fn execute(self, catalog: &dyn Catalog) -> Result<RemoveDanglingDeleteFilesResult> {
        // An unpartitioned single-spec table has nothing to do. Every commit's ManifestFilterManager
        // already drops table-wide-applicable deletes.
        let metadata = self.table.metadata();
        if metadata.partition_specs_iter().count() == 1
            && metadata.default_partition_spec().is_unpartitioned()
        {
            return Ok(RemoveDanglingDeleteFilesResult::default());
        }

        // The action scopes to the current snapshot only.
        let Some(snapshot) = metadata.current_snapshot().cloned() else {
            return Ok(RemoveDanglingDeleteFilesResult::default());
        };

        let live = self.collect_live_entries(&snapshot).await?;
        let dangling = find_dangling_deletes(&live);

        if dangling.is_empty() {
            // An empty plan commits nothing. Java also commits only on a non-empty dangling set.
            return Ok(RemoveDanglingDeleteFilesResult::default());
        }

        // One RewriteFiles delete-file removal. The recorded operation is Replace.
        let transaction = Transaction::new(&self.table);
        let action = transaction
            .rewrite_files(Vec::new(), Vec::new())
            .delete_delete_files(dangling.clone());
        let transaction = action.apply(transaction)?;
        transaction.commit(catalog).await?;

        Ok(RemoveDanglingDeleteFilesResult {
            removed_delete_files: dangling,
        })
    }

    /// Walk the current snapshot's manifests once and collect every live entry into a [`LiveEntries`]
    /// view.
    async fn collect_live_entries(&self, snapshot: &crate::spec::Snapshot) -> Result<LiveEntries> {
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
                        // A live DATA entry: contribute its (post-inheritance) data sequence number to its
                        // partition+spec minimum, and record its path (for the DV reference check).
                        let key = (data_file.partition_spec_id, data_file.partition().clone());
                        let seq = entry.sequence_number();
                        live.min_data_seq_by_group
                            .entry(key)
                            .and_modify(|current| *current = min_option(*current, seq))
                            .or_insert(seq);
                        live.live_data_file_paths
                            .insert(data_file.file_path().to_string());
                    }
                    DataContentType::PositionDeletes | DataContentType::EqualityDeletes => {
                        live.live_delete_entries.push(LiveDeleteEntry {
                            data_file: data_file.clone(),
                            sequence_number: entry.sequence_number(),
                        });
                    }
                }
            }
        }

        Ok(live)
    }
}

/// The grouping key. Java groups and joins on the spec id and the partition tuple together.
type GroupKey = (i32, Struct);

/// A live delete entry and its post-inheritance data sequence number.
struct LiveDeleteEntry {
    data_file: DataFile,
    sequence_number: Option<i64>,
}

/// The live-entry view the dangling predicate runs over.
#[derive(Default)]
struct LiveEntries {
    /// `(spec_id, partition) -> min(data_sequence_number)` over live data files. An absent group means
    /// the partition and spec hold no live data file. That is the `min IS NULL` dangling case.
    min_data_seq_by_group: HashMap<GroupKey, Option<i64>>,
    /// Every live data file's path. The deletion-vector reference check matches against this set.
    live_data_file_paths: HashSet<String>,
    /// The live delete entries (position / equality / DV).
    live_delete_entries: Vec<LiveDeleteEntry>,
}

/// Take the minimum of two sequence numbers, where `None` means "not yet set": `min(None, x) = x`.
///
/// # Notes
///
/// Real on-disk entries always carry a sequence number, so `None` only occurs on the first insert.
fn min_option(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    match (left, right) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}

/// Return every delete file that can no longer apply to any live data file.
///
/// # Notes
///
/// This is the pure core of the action. It does no IO, so the off-by-one corruption edge is
/// unit-testable directly. Java splits it across `findDanglingDeletes` and `findDanglingDvs`.
fn find_dangling_deletes(live: &LiveEntries) -> Vec<DataFile> {
    let mut dangling = Vec::new();

    for entry in &live.live_delete_entries {
        let data_file = &entry.data_file;

        // A file-scoped delete dangles when its referenced path is not a live data-file path. The read
        // path routes such a delete BY PATH in `crate::delete_file_index`, and that lookup consults
        // neither the spec id nor the partition tuple. The per-partition min-seq rule therefore says
        // nothing about whether it still applies. This check must come first. Judging a file-scoped
        // delete by the min-seq rule removes a delete the reader still honors, and the masked rows
        // resurrect permanently, because the removal is a committed metadata change.
        if let Some(referenced) = referenced_data_file_location(data_file) {
            if !live.live_data_file_paths.contains(&referenced) {
                dangling.push(data_file.clone());
            }
            continue;
        }
        // A deletion vector with no `referenced_data_file` is malformed. Its reference can never match a
        // live path, so it dangles by absence. This matches the left-outer-join-then-null semantics.
        if is_deletion_vector(data_file) {
            dangling.push(data_file.clone());
            continue;
        }

        // A parquet position or equality delete compares against its group's minimum live data sequence
        // number.
        let key = (data_file.partition_spec_id, data_file.partition().clone());
        match live.min_data_seq_by_group.get(&key) {
            // The group holds no live data file, so the delete applies to nothing. It dangles whatever
            // its content type and sequence number are.
            None | Some(None) => dangling.push(data_file.clone()),
            Some(Some(min_data_seq)) => {
                let delete_seq = entry.sequence_number;
                let is_dangling = match data_file.content_type() {
                    // Strict `<`. A position delete at the exact minimum still applies, because the read
                    // path uses `delete_seq >= data_seq`.
                    DataContentType::PositionDeletes => {
                        delete_seq.is_some_and(|seq| seq < *min_data_seq)
                    }
                    // Non-strict `<=`. An equality delete applies only to strictly lower-sequence data,
                    // so one at the exact minimum does not apply, and it dangles.
                    DataContentType::EqualityDeletes => {
                        delete_seq.is_some_and(|seq| seq <= *min_data_seq)
                    }
                    DataContentType::Data => false,
                };
                if is_dangling {
                    dangling.push(data_file.clone());
                }
            }
        }
    }

    dangling
}

/// Report whether a delete file is a deletion vector, which means a Puffin-format position delete.
///
/// # Notes
///
/// A deletion vector is file-scoped, so it follows the reference rule, not the min-seq rule.
fn is_deletion_vector(data_file: &DataFile) -> bool {
    data_file.content_type() == DataContentType::PositionDeletes
        && data_file.file_format() == DataFileFormat::Puffin
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use futures::TryStreamExt;
    use tempfile::TempDir;

    use super::*;
    use crate::io::LocalFsStorageFactory;
    use crate::memory::MemoryCatalogBuilder;
    use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, FormatVersion, Literal,
        ManifestStatus, NestedField, Operation, PartitionSpec, PrimitiveType, Schema, Struct,
        Transform, Type,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::writer::base_writer::equality_delete_writer::{
        EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
    };
    use crate::writer::base_writer::position_delete_writer::{
        PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
        position_delete_writer_properties,
    };
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};
    use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

    // =========================================================================================
    // Pure-fn tests (the dangling predicate — the off-by-one corruption edge, IO-free)
    // =========================================================================================

    /// Build a synthetic delete [`DataFile`] of the given content / format in partition `x = part`.
    fn delete_file(
        path: &str,
        content: DataContentType,
        format: DataFileFormat,
        part: i64,
        spec_id: i32,
    ) -> DataFile {
        let mut builder = DataFileBuilder::default();
        builder
            .content(content)
            .file_path(path.to_string())
            .file_format(format)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(spec_id)
            .partition(Struct::from_iter([Some(Literal::long(part))]));
        if content == DataContentType::EqualityDeletes {
            builder.equality_ids(Some(vec![2]));
        }
        if format == DataFileFormat::Puffin {
            builder
                .content_offset(Some(4))
                .content_size_in_bytes(Some(40))
                .referenced_data_file(Some("placeholder.parquet".to_string()));
        }
        builder.build().unwrap()
    }

    /// Build a `LiveEntries` view with the given per-group data minimums and live data paths.
    fn live_entries(
        data_min_by_group: &[((i32, i64), Option<i64>)],
        data_paths: &[&str],
        deletes: Vec<LiveDeleteEntry>,
    ) -> LiveEntries {
        let mut live = LiveEntries::default();
        for ((spec_id, part), min) in data_min_by_group {
            live.min_data_seq_by_group.insert(
                (*spec_id, Struct::from_iter([Some(Literal::long(*part))])),
                *min,
            );
        }
        for path in data_paths {
            live.live_data_file_paths.insert((*path).to_string());
        }
        live.live_delete_entries = deletes;
        live
    }

    /// The off-by-one pin. With a partition minimum data sequence of 5:
    ///
    /// | Delete | Seq | Dangling |
    /// |---|---|---|
    /// | position | 5 | no |
    /// | position | 4 | yes |
    /// | equality | 5 | yes |
    /// | equality | 6 | no |
    ///
    /// Mutation: flip the position `<` to `<=` and the same-sequence position delete is wrongly
    /// removed, which resurrects rows. Flip the equality `<=` to `<` and the same-sequence equality
    /// delete is wrongly kept. Both directions fail this test.
    #[test]
    fn test_dangling_predicate_off_by_one_between_position_and_equality() {
        let deletes = vec![
            LiveDeleteEntry {
                data_file: delete_file(
                    "pos-at-min.parquet",
                    DataContentType::PositionDeletes,
                    DataFileFormat::Parquet,
                    0,
                    0,
                ),
                sequence_number: Some(5),
            },
            LiveDeleteEntry {
                data_file: delete_file(
                    "pos-below-min.parquet",
                    DataContentType::PositionDeletes,
                    DataFileFormat::Parquet,
                    0,
                    0,
                ),
                sequence_number: Some(4),
            },
            LiveDeleteEntry {
                data_file: delete_file(
                    "eq-at-min.parquet",
                    DataContentType::EqualityDeletes,
                    DataFileFormat::Parquet,
                    0,
                    0,
                ),
                sequence_number: Some(5),
            },
            LiveDeleteEntry {
                data_file: delete_file(
                    "eq-above-min.parquet",
                    DataContentType::EqualityDeletes,
                    DataFileFormat::Parquet,
                    0,
                    0,
                ),
                sequence_number: Some(6),
            },
        ];
        let live = live_entries(&[((0, 0), Some(5))], &["data.parquet"], deletes);

        let dangling: HashSet<String> = find_dangling_deletes(&live)
            .into_iter()
            .map(|file| file.file_path().to_string())
            .collect();

        assert_eq!(
            dangling,
            HashSet::from([
                "pos-below-min.parquet".to_string(),
                "eq-at-min.parquet".to_string(),
            ]),
            "pos dangles strictly below min (NOT at min); eq dangles at-or-below min"
        );
    }

    /// A delete whose group holds no live data file dangles whatever its content type and sequence
    /// number. This pins the join-miss branch.
    #[test]
    fn test_dangling_when_no_live_data_in_partition() {
        let deletes = vec![
            LiveDeleteEntry {
                data_file: delete_file(
                    "pos-orphan.parquet",
                    DataContentType::PositionDeletes,
                    DataFileFormat::Parquet,
                    9,
                    0,
                ),
                sequence_number: Some(100),
            },
            LiveDeleteEntry {
                data_file: delete_file(
                    "eq-orphan.parquet",
                    DataContentType::EqualityDeletes,
                    DataFileFormat::Parquet,
                    9,
                    0,
                ),
                sequence_number: Some(100),
            },
        ];
        // The data minimum exists for partition 0 ONLY; the deletes are in partition 9 (no live data).
        let live = live_entries(&[((0, 0), Some(5))], &["data.parquet"], deletes);

        let dangling: HashSet<String> = find_dangling_deletes(&live)
            .into_iter()
            .map(|file| file.file_path().to_string())
            .collect();
        assert_eq!(
            dangling,
            HashSet::from([
                "pos-orphan.parquet".to_string(),
                "eq-orphan.parquet".to_string(),
            ]),
            "a delete with no live data file in its partition+spec always dangles"
        );
    }

    /// Cross-spec isolation. One partition tuple under two spec ids does not share a minimum. A delete
    /// under spec 1 compares only against spec-1 data, so it dangles when only spec-0 data is live.
    #[test]
    fn test_cross_spec_grouping_does_not_share_minimum() {
        let deletes = vec![LiveDeleteEntry {
            data_file: delete_file(
                "spec1-pos.parquet",
                DataContentType::PositionDeletes,
                DataFileFormat::Parquet,
                0,
                1,
            ),
            sequence_number: Some(100),
        }];
        // Live data minimum exists for (spec 0, part 0) only; the delete is (spec 1, part 0).
        let live = live_entries(&[((0, 0), Some(5))], &["data.parquet"], deletes);

        let dangling: Vec<String> = find_dangling_deletes(&live)
            .into_iter()
            .map(|file| file.file_path().to_string())
            .collect();
        assert_eq!(
            dangling,
            vec!["spec1-pos.parquet".to_string()],
            "a spec-1 delete must not borrow spec-0's data minimum"
        );
    }

    /// The irreversible-delete pin. The read path routes a file-scoped position delete BY PATH, with no
    /// spec and no partition condition. Calling it dangling because its own group holds no live data
    /// file removes a delete the reader still honors, and the masked rows resurrect permanently.
    ///
    /// A non-file-scoped delete in the same empty group must still dangle. The rule is "route by
    /// reference", not "stop collecting".
    ///
    /// Mutation: drop the file-scoped leg and both file-scoped deletes enter the dangling set. Drop the
    /// `contains` check and `pos-file-scoped-gone.parquet` is never collected. Both turn this test red.
    #[test]
    fn test_file_scoped_position_delete_referencing_live_data_is_not_dangling() {
        let referenced_field = {
            let mut file = delete_file(
                "pos-file-scoped-field.parquet",
                DataContentType::PositionDeletes,
                DataFileFormat::Parquet,
                9,
                0,
            );
            file.referenced_data_file = Some("live-data.parquet".to_string());
            file
        };
        let referenced_bounds = {
            let mut file = delete_file(
                "pos-file-scoped-bounds.parquet",
                DataContentType::PositionDeletes,
                DataFileFormat::Parquet,
                9,
                0,
            );
            file.lower_bounds = HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("live-data.parquet"),
            )]);
            file.upper_bounds = HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("live-data.parquet"),
            )]);
            file
        };
        let referenced_gone = {
            let mut file = delete_file(
                "pos-file-scoped-gone.parquet",
                DataContentType::PositionDeletes,
                DataFileFormat::Parquet,
                9,
                0,
            );
            file.referenced_data_file = Some("rewritten-away.parquet".to_string());
            file
        };
        let partition_scoped = delete_file(
            "pos-partition-scoped.parquet",
            DataContentType::PositionDeletes,
            DataFileFormat::Parquet,
            9,
            0,
        );

        let deletes = vec![
            LiveDeleteEntry {
                data_file: referenced_field,
                sequence_number: Some(2),
            },
            LiveDeleteEntry {
                data_file: referenced_bounds,
                sequence_number: Some(2),
            },
            LiveDeleteEntry {
                data_file: referenced_gone,
                sequence_number: Some(2),
            },
            LiveDeleteEntry {
                data_file: partition_scoped,
                sequence_number: Some(2),
            },
        ];
        // Live data exists ONLY in (spec 0, partition 0); every delete above is stamped partition 9,
        // so the partition min-seq rule would call all four dangling.
        let live = live_entries(&[((0, 0), Some(1))], &["live-data.parquet"], deletes);

        let dangling: HashSet<String> = find_dangling_deletes(&live)
            .into_iter()
            .map(|file| file.file_path().to_string())
            .collect();
        assert_eq!(
            dangling,
            HashSet::from([
                "pos-file-scoped-gone.parquet".to_string(),
                "pos-partition-scoped.parquet".to_string(),
            ]),
            "file-scoped deletes dangle by REFERENCE (only the one naming a gone data file), while a \
             partition-scoped delete in the same empty group still dangles by the min-seq rule"
        );
    }

    /// An equality delete is never file-scoped, whatever bounds it carries. Java's
    /// `ContentFileUtil.referencedDataFile` returns null for equality deletes before it reads the
    /// bounds. The min-seq rule must judge it. Both directions:
    ///
    /// - it dangles when its partition holds no live data, even though its bounds name a live file;
    /// - it does not dangle while it still applies, even though its bounds name a file that is gone.
    ///   Removing it would resurrect the rows it masks, and the removal is irreversible.
    ///
    /// Mutation: drop the equality early return in `referenced_data_file_location` and both flip. No
    /// read-path test catches the second case, because the index routes by content type first.
    #[test]
    fn test_equality_delete_with_path_bounds_is_judged_by_min_seq_not_by_reference() {
        let path_bounds = |path: &str| {
            HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string(path.to_string()),
            )])
        };

        // In partition 9 — no live data there — with bounds naming a LIVE data file.
        let orphan_eq = {
            let mut file = delete_file(
                "eq-orphan-bounds-live.parquet",
                DataContentType::EqualityDeletes,
                DataFileFormat::Parquet,
                9,
                0,
            );
            file.lower_bounds = path_bounds("live-data.parquet");
            file.upper_bounds = path_bounds("live-data.parquet");
            file
        };
        // In partition 0 at seq 5 — still applies over the min of 1 — with bounds naming a GONE file.
        let applicable_eq = {
            let mut file = delete_file(
                "eq-applicable-bounds-gone.parquet",
                DataContentType::EqualityDeletes,
                DataFileFormat::Parquet,
                0,
                0,
            );
            file.lower_bounds = path_bounds("rewritten-away.parquet");
            file.upper_bounds = path_bounds("rewritten-away.parquet");
            file
        };

        let deletes = vec![
            LiveDeleteEntry {
                data_file: orphan_eq,
                sequence_number: Some(5),
            },
            LiveDeleteEntry {
                data_file: applicable_eq,
                sequence_number: Some(5),
            },
        ];
        let live = live_entries(&[((0, 0), Some(1))], &["live-data.parquet"], deletes);

        let dangling: Vec<String> = find_dangling_deletes(&live)
            .into_iter()
            .map(|file| file.file_path().to_string())
            .collect();
        assert_eq!(
            dangling,
            vec!["eq-orphan-bounds-live.parquet".to_string()],
            "equality deletes follow the partition min-seq rule; their file_path bounds are not a \
             file scope"
        );
    }

    /// A deletion vector dangles when its reference is not a live data-file path, and does not dangle
    /// when the reference is live. This pins both directions of the reference branch.
    #[test]
    fn test_dv_dangles_when_referenced_data_file_gone() {
        let live_dv = {
            let mut file = delete_file(
                "live-dv.puffin",
                DataContentType::PositionDeletes,
                DataFileFormat::Puffin,
                0,
                0,
            );
            file.referenced_data_file = Some("live-data.parquet".to_string());
            file
        };
        let dangling_dv = {
            let mut file = delete_file(
                "dangling-dv.puffin",
                DataContentType::PositionDeletes,
                DataFileFormat::Puffin,
                0,
                0,
            );
            file.referenced_data_file = Some("gone-data.parquet".to_string());
            file
        };
        let deletes = vec![
            LiveDeleteEntry {
                data_file: live_dv,
                sequence_number: Some(2),
            },
            LiveDeleteEntry {
                data_file: dangling_dv,
                sequence_number: Some(2),
            },
        ];
        let live = live_entries(&[((0, 0), Some(1))], &["live-data.parquet"], deletes);

        let dangling: Vec<String> = find_dangling_deletes(&live)
            .into_iter()
            .map(|file| file.file_path().to_string())
            .collect();
        assert_eq!(
            dangling,
            vec!["dangling-dv.puffin".to_string()],
            "a DV referencing a gone data file dangles; one referencing a live file does not"
        );
    }

    // =========================================================================================
    // End-to-end tests. Real parquet, real delete writers, real scans, real commits.
    // =========================================================================================

    async fn local_fs_catalog() -> (impl Catalog, TempDir) {
        let temp_dir = TempDir::new().expect("temp dir");
        let warehouse = temp_dir
            .path()
            .to_str()
            .expect("utf8 temp path")
            .to_string();
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "memory",
                std::collections::HashMap::from([("warehouse".to_string(), warehouse)]),
            )
            .await
            .expect("load local-fs memory catalog");
        (catalog, temp_dir)
    }

    fn three_long_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                Arc::new(NestedField::required(
                    1,
                    "x",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::required(
                    2,
                    "y",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::required(
                    3,
                    "z",
                    Type::Primitive(PrimitiveType::Long),
                )),
            ])
            .build()
            .expect("build schema")
    }

    async fn create_partitioned_table(
        catalog: &impl Catalog,
        format_version: FormatVersion,
    ) -> Table {
        let schema = three_long_schema();
        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_partition_field("x", "x", Transform::Identity)
            .expect("add partition field")
            .build()
            .expect("build spec");
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog
            .create_namespace(&namespace, std::collections::HashMap::new())
            .await
            .expect("create namespace");
        let table_ident = TableIdent::new(namespace.clone(), "t".to_string());
        let creation = TableCreation::builder()
            .name(table_ident.name().to_string())
            .schema(schema)
            .partition_spec(spec)
            .format_version(format_version)
            .build();
        catalog
            .create_table(&namespace, creation)
            .await
            .expect("create table")
    }

    async fn write_data_file(
        table: &Table,
        file_name: &str,
        part_value: i64,
        rows: &[(i64, i64, i64)],
    ) -> DataFile {
        use crate::arrow::schema_to_arrow_schema;

        let schema = table.metadata().current_schema();
        let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());

        let xs: Vec<i64> = rows.iter().map(|(x, _, _)| *x).collect();
        let ys: Vec<i64> = rows.iter().map(|(_, y, _)| *y).collect();
        let zs: Vec<i64> = rows.iter().map(|(_, _, z)| *z).collect();
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int64Array::from(xs)) as ArrayRef,
            Arc::new(Int64Array::from(ys)) as ArrayRef,
            Arc::new(Int64Array::from(zs)) as ArrayRef,
        ])
        .unwrap();

        let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
        let output = table.file_io().new_output(file_path).unwrap();
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            schema.clone(),
        );
        let mut writer = parquet_builder.build(output).await.unwrap();
        writer.write(&batch).await.unwrap();
        let data_file_builders = writer.close().await.unwrap();

        let mut builder = data_file_builders.into_iter().next().unwrap();
        builder
            .content(DataContentType::Data)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .build()
            .unwrap()
    }

    async fn write_equality_delete_file(
        table: &Table,
        part_value: i64,
        delete_ys: &[i64],
    ) -> DataFile {
        use crate::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};

        let schema = table.metadata().current_schema().clone();
        let config = EqualityDeleteWriterConfig::new(vec![2], schema.clone()).unwrap();
        let delete_schema =
            Arc::new(arrow_schema_to_schema(config.projected_arrow_schema_ref()).unwrap());

        let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
        let file_name_gen = DefaultFileNameGenerator::new(
            "eq-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            delete_schema,
        );
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );

        let partition_key = crate::spec::PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            schema.clone(),
            Struct::from_iter([Some(Literal::long(part_value))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
            .build(Some(partition_key))
            .await
            .unwrap();

        let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let xs: Vec<i64> = delete_ys.iter().map(|_| part_value).collect();
        let ys: Vec<i64> = delete_ys.to_vec();
        let zs: Vec<i64> = delete_ys.iter().map(|_| 0).collect();
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int64Array::from(xs)) as ArrayRef,
            Arc::new(Int64Array::from(ys)) as ArrayRef,
            Arc::new(Int64Array::from(zs)) as ArrayRef,
        ])
        .unwrap();
        writer.write(batch).await.unwrap();
        writer.close().await.unwrap().into_iter().next().unwrap()
    }

    async fn write_position_delete_file(
        table: &Table,
        part_value: i64,
        deletes: &[(String, i64)],
    ) -> DataFile {
        let config = PositionDeleteWriterConfig::new().unwrap();
        let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
        let file_name_gen = DefaultFileNameGenerator::new(
            "pos-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            config.schema().clone(),
        );
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );
        let partition_key = crate::spec::PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::long(part_value))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
            .build(Some(partition_key))
            .await
            .unwrap();

        let paths: Vec<&str> = deletes.iter().map(|(path, _)| path.as_str()).collect();
        let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
        let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(Int64Array::from(positions)) as ArrayRef,
        ])
        .unwrap();
        writer.write(batch).await.unwrap();
        writer.close().await.unwrap().into_iter().next().unwrap()
    }

    /// Write a real parquet position delete with [`MetricsConfig::for_position_delete`]. That config
    /// forces the reserved `file_path` column to full metrics, so the file carries equal `file_path`
    /// bounds and is file-scoped, as a Java-written one is. Java's `PositionDeleteWriter` never sets
    /// `referenced_data_file`; it preserves those bounds instead.
    ///
    /// `part_value` is the partition tuple the delete is stamped with. Callers pick one that differs
    /// from the referenced data file's partition.
    async fn write_file_scoped_position_delete_file(
        table: &Table,
        part_value: i64,
        deletes: &[(String, i64)],
    ) -> DataFile {
        use crate::spec::MetricsConfig;

        let config = PositionDeleteWriterConfig::new().unwrap();
        let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
        let file_name_gen = DefaultFileNameGenerator::new(
            "file-scoped-pos-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        // position_delete_writer_properties() disables the 64-byte parquet stats truncate, so the
        // file_path bounds stay exact. MetricsConfig::for_position_delete keeps Full mode.
        let parquet_builder =
            ParquetWriterBuilder::new(position_delete_writer_properties(), config.schema().clone())
                .with_metrics_config(MetricsConfig::for_position_delete());
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );
        let partition_key = crate::spec::PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::long(part_value))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
            .build(Some(partition_key))
            .await
            .unwrap();

        let paths: Vec<&str> = deletes.iter().map(|(path, _)| path.as_str()).collect();
        let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
        let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(Int64Array::from(positions)) as ArrayRef,
        ])
        .unwrap();
        writer.write(batch).await.unwrap();
        writer.close().await.unwrap().into_iter().next().unwrap()
    }

    async fn append_files(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let action = tx.fast_append().add_data_files(files);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    async fn add_deletes(catalog: &impl Catalog, table: &Table, deletes: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let action = tx.row_delta().add_deletes(deletes);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Write a real Puffin deletion vector referencing `data_file_path` at the given positions.
    /// Deletion vectors require format version 3.
    async fn write_real_dv_file(
        table: &Table,
        file_name: &str,
        data_file_path: &str,
        part_value: i64,
        positions: &[u64],
    ) -> DataFile {
        use crate::spec::PartitionKey;
        use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;

        let partition_key = PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::long(part_value))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let dv_path = format!("{}/data/{}", table.metadata().location(), file_name);
        let output_file = table.file_io().new_output(&dv_path).unwrap();
        let mut dv_writer = DVFileWriter::new(output_file);
        for pos in positions {
            dv_writer
                .delete(data_file_path, *pos, Some(&partition_key))
                .unwrap();
        }
        dv_writer.close().await.unwrap().into_iter().next().unwrap()
    }

    /// Scan the table and collect the `y` column values with merge-on-read deletes applied.
    async fn scan_y_values(table: &Table) -> HashSet<i64> {
        let stream = table
            .scan()
            .select(["y"])
            .build()
            .unwrap()
            .to_arrow()
            .await
            .unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let mut values = HashSet::new();
        for batch in batches {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for index in 0..col.len() {
                values.insert(col.value(index));
            }
        }
        values
    }

    /// The set of live delete-file paths in the current snapshot.
    async fn live_delete_paths(table: &Table) -> HashSet<String> {
        use crate::spec::ManifestContentType;
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut paths = HashSet::new();
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != ManifestContentType::Deletes {
                continue;
            }
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.is_alive() {
                    paths.insert(entry.file_path().to_string());
                }
            }
        }
        paths
    }

    fn summary_prop(table: &Table, prop: &str) -> Option<String> {
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties
            .get(prop)
            .cloned()
    }

    /// The resurrection door. A still-applicable equality delete must survive the action, and the scan
    /// must stay correct afterwards. Append X at sequence 1, then an equality delete at sequence 2
    /// removing y=20. The delete applies, so the action removes nothing and the scan still drops y=20.
    ///
    /// Mutation: flipping `<=` to `<` still keeps this delete. Reversing the comparison to `seq >= min`
    /// is the real resurrection lever, and the off-by-one pure-function test catches that.
    #[tokio::test]
    async fn test_crown_jewel_still_applicable_equality_delete_not_removed() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

        let x = write_data_file(&table, "x.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let table = append_files(&catalog, &table, vec![x]).await;

        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let eq_path = eq_delete.file_path().to_string();
        let table = add_deletes(&catalog, &table, vec![eq_delete]).await;
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30]),
            "before the action the equality delete drops y=20"
        );

        let result = RemoveDanglingDeleteFiles::new(table.clone())
            .execute(&catalog)
            .await
            .unwrap();
        assert!(
            result.removed_delete_files.is_empty(),
            "a still-applicable equality delete must NOT be removed: {:?}",
            result.removed_delete_files
        );

        // No snapshot was committed (empty plan) — the table head is unchanged, the delete still live.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            live_delete_paths(&reloaded).await.contains(&eq_path),
            "the applicable equality delete must still be live"
        );
        assert_eq!(
            scan_y_values(&reloaded).await,
            HashSet::from([10, 30]),
            "the scan must still drop y=20 (no resurrection)"
        );
    }

    /// The position-delete exact-sequence boundary, end to end. Append X at sequence 1, then a position
    /// delete at sequence 2. Compact X to X' preserving sequence 1. The position delete is
    /// partition-scoped, so it compares against the partition minimum of 1. Sequence 2 is not below 1,
    /// so the action keeps it. This pins that a position delete at or above the minimum survives.
    #[tokio::test]
    async fn test_position_delete_at_or_above_partition_min_is_not_dangling() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let x_path = x.file_path().to_string();
        let table = append_files(&catalog, &table, vec![x]).await;
        let x_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        let pos_delete = write_position_delete_file(&table, 0, &[(x_path, 1)]).await;
        let pos_path = pos_delete.file_path().to_string();
        let table = add_deletes(&catalog, &table, vec![pos_delete]).await;
        let delete_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();
        assert!(x_seq < delete_seq, "delete is at a higher seq than X");

        // A second data file Y in the SAME partition at a HIGHER seq keeps the partition populated.
        let y = write_data_file(&table, "y.parquet", 0, &[(0, 40, 400)]).await;
        let table = append_files(&catalog, &table, vec![y]).await;

        // The partition-0 minimum data seq is still x_seq (X is live). The position delete (delete_seq)
        // is NOT < x_seq, so it is NOT dangling.
        let result = RemoveDanglingDeleteFiles::new(table.clone())
            .execute(&catalog)
            .await
            .unwrap();
        assert!(
            result.removed_delete_files.is_empty(),
            "a position delete at/above the partition minimum must not be removed: {:?}",
            result.removed_delete_files
        );
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            live_delete_paths(&reloaded).await.contains(&pos_path),
            "the position delete must still be live"
        );
    }

    /// A genuinely dangling equality delete is removed, the scan is unchanged, and the counter is
    /// correct. Append X at sequence 1 and delete y=20 at sequence 2. Compact X to X' with a fresh
    /// higher sequence, so the partition minimum jumps to 3 and the delete dangles. The scan does not
    /// change, because the fresh-sequence rewrite already stopped the delete from applying.
    #[tokio::test]
    async fn test_genuinely_dangling_equality_delete_is_removed_with_counter() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

        let x = write_data_file(&table, "x.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let table = append_files(&catalog, &table, vec![x.clone()]).await;

        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let eq_path = eq_delete.file_path().to_string();
        let table = add_deletes(&catalog, &table, vec![eq_delete]).await;

        // Compact X→X' with a FRESH higher seq (no preservation) — the equality delete stops applying.
        let x_prime = write_data_file(&table, "x-prime.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![x], vec![x_prime]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 20, 30]),
            "the fresh-seq rewrite already let y=20 back (the delete dangles)"
        );

        let result = RemoveDanglingDeleteFiles::new(table.clone())
            .execute(&catalog)
            .await
            .unwrap();
        assert_eq!(
            result.removed_delete_files.len(),
            1,
            "the dangling equality delete must be removed"
        );
        assert_eq!(result.removed_equality_delete_files_count(), 1);

        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            !live_delete_paths(&reloaded).await.contains(&eq_path),
            "the dangling equality delete must be tombstoned"
        );
        assert_eq!(
            reloaded
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Replace,
            "the removal commits a Replace snapshot (Java BaseRewriteFiles.operation)"
        );
        assert_eq!(
            summary_prop(&reloaded, "removed-equality-delete-files").as_deref(),
            Some("1"),
            "the summary must report one removed equality delete"
        );
        // The scan is unchanged by the GC (the delete was already not applying).
        assert_eq!(
            scan_y_values(&reloaded).await,
            HashSet::from([10, 20, 30]),
            "the GC does not change the read result"
        );
    }

    /// A dangling parquet position delete is removed. A plain RewriteFiles keeps it, so this action is
    /// the cleaner. Append X at sequence 1 and a position delete at sequence 2. Compact X to X' with a
    /// fresh sequence 3, so the partition minimum jumps to 3 and the position delete dangles.
    #[tokio::test]
    async fn test_dangling_position_delete_parquet_removed_after_data_rewritten_away() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

        let x = write_data_file(&table, "x.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let x_path = x.file_path().to_string();
        let table = append_files(&catalog, &table, vec![x.clone()]).await;

        let pos_delete = write_position_delete_file(&table, 0, &[(x_path, 1)]).await;
        let pos_path = pos_delete.file_path().to_string();
        let table = add_deletes(&catalog, &table, vec![pos_delete]).await;
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30]),
            "the position delete drops y=20"
        );

        // Compact X→X' FRESH seq — the position delete now references a gone file and the partition min
        // jumps above the delete's seq, so it dangles.
        let x_prime = write_data_file(&table, "x-prime.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![x], vec![x_prime]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // A plain RewriteFiles KEEPS the dangling position delete (carry-unchanged posture).
        assert!(
            live_delete_paths(&table).await.contains(&pos_path),
            "plain RewriteFiles keeps the now-dangling position delete (carry posture)"
        );

        let result = RemoveDanglingDeleteFiles::new(table.clone())
            .execute(&catalog)
            .await
            .unwrap();
        assert_eq!(result.removed_delete_files.len(), 1);
        assert_eq!(result.removed_position_delete_files_count(), 1);

        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            !live_delete_paths(&reloaded).await.contains(&pos_path),
            "the dangling position delete must be removed by this action"
        );
        assert_eq!(
            summary_prop(&reloaded, "removed-position-delete-files").as_deref(),
            Some("1"),
            "the summary must report one removed position delete"
        );
        // The scan is unchanged (y=20 was already back after the fresh-seq rewrite).
        assert_eq!(
            scan_y_values(&reloaded).await,
            HashSet::from([10, 20, 30]),
            "the GC does not change the read result"
        );
    }

    /// Partition isolation, end to end. Partition 0 holds X at sequence 1 and an applicable equality
    /// delete at sequence 2. Partition 1's data is rewritten to a fresh higher sequence, so its equality
    /// delete dangles. The action removes only the partition-1 delete.
    #[tokio::test]
    async fn test_partition_isolation_dangling_in_one_applicable_in_other() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

        // Partition 0: X0 (seq 1) + applicable equality delete (seq 2).
        let x0 = write_data_file(&table, "x0.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        // Partition 1: X1 (seq 1).
        let x1 = write_data_file(&table, "x1.parquet", 1, &[(1, 60, 600), (1, 70, 700)]).await;
        let table = append_files(&catalog, &table, vec![x0, x1.clone()]).await;

        let eq0 = write_equality_delete_file(&table, 0, &[20]).await;
        let eq0_path = eq0.file_path().to_string();
        let eq1 = write_equality_delete_file(&table, 1, &[70]).await;
        let eq1_path = eq1.file_path().to_string();
        let table = add_deletes(&catalog, &table, vec![eq0, eq1]).await;

        // Rewrite partition 1's data X1→X1' with a FRESH higher seq, so eq1 (partition 1) dangles while
        // eq0 (partition 0, applies to the still-seq-1 X0) stays applicable.
        let x1_prime =
            write_data_file(&table, "x1-prime.parquet", 1, &[(1, 60, 600), (1, 70, 700)]).await;
        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![x1], vec![x1_prime]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let result = RemoveDanglingDeleteFiles::new(table.clone())
            .execute(&catalog)
            .await
            .unwrap();
        let removed: HashSet<String> = result
            .removed_delete_files
            .iter()
            .map(|file| file.file_path().to_string())
            .collect();
        assert_eq!(
            removed,
            HashSet::from([eq1_path.clone()]),
            "only the partition-1 dangling delete is removed; the partition-0 delete is kept"
        );

        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        let live = live_delete_paths(&reloaded).await;
        assert!(
            live.contains(&eq0_path),
            "the applicable partition-0 delete stays live"
        );
        assert!(
            !live.contains(&eq1_path),
            "the dangling partition-1 delete is gone"
        );
        // Partition 0's scan still drops y=20 (its delete still applies — no resurrection).
        assert!(
            !scan_y_values(&reloaded).await.contains(&20),
            "the applicable partition-0 delete must still drop y=20"
        );
    }

    /// A table whose deletes all still apply commits nothing, and the snapshot id does not change. This
    /// pins that the action never mints an empty Replace snapshot.
    #[tokio::test]
    async fn test_empty_plan_is_a_no_op_no_commit() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x]).await;
        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let table = add_deletes(&catalog, &table, vec![eq_delete]).await;
        let head_before = table.metadata().current_snapshot().unwrap().snapshot_id();

        let result = RemoveDanglingDeleteFiles::new(table.clone())
            .execute(&catalog)
            .await
            .unwrap();
        assert!(result.removed_delete_files.is_empty());

        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            reloaded
                .metadata()
                .current_snapshot()
                .unwrap()
                .snapshot_id(),
            head_before,
            "no snapshot must be committed for an empty plan"
        );
    }

    /// A table with NO current snapshot is a no-op (defensive — nothing to scan).
    #[tokio::test]
    async fn test_no_current_snapshot_is_a_no_op() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

        let result = RemoveDanglingDeleteFiles::new(table)
            .execute(&catalog)
            .await
            .unwrap();
        assert!(result.removed_delete_files.is_empty());
    }

    /// The removed delete file must be tombstoned in the rewritten DELETE manifest, not only reported in
    /// the result. Only then does the read path stop applying it.
    ///
    /// Mutation: make `with_removed_delete_files` inert and the delete stays live, which fails here.
    #[tokio::test]
    async fn test_removed_delete_is_tombstoned_in_rewritten_manifest() {
        use crate::spec::ManifestContentType;
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x.clone()]).await;
        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let eq_path = eq_delete.file_path().to_string();
        let table = add_deletes(&catalog, &table, vec![eq_delete]).await;

        // Rewrite X away to a fresh seq so the equality delete dangles.
        let x_prime =
            write_data_file(&table, "x-prime.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![x], vec![x_prime]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        RemoveDanglingDeleteFiles::new(table.clone())
            .execute(&catalog)
            .await
            .unwrap();

        // The removed delete must appear as a DELETED tombstone in a DELETE manifest of the new snapshot.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        let snapshot = reloaded.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(reloaded.file_io(), reloaded.metadata())
            .await
            .unwrap();
        let mut found_tombstone = false;
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != ManifestContentType::Deletes {
                continue;
            }
            let manifest = manifest_file
                .load_manifest(reloaded.file_io())
                .await
                .unwrap();
            for entry in manifest.entries() {
                if entry.file_path() == eq_path && entry.status() == ManifestStatus::Deleted {
                    found_tombstone = true;
                }
            }
        }
        assert!(
            found_tombstone,
            "the dangling equality delete must be a Deleted tombstone (producer routing fired)"
        );
    }

    /// The irreversible-delete pin, end to end through real parquet metrics.
    ///
    /// A position delete written with [`MetricsConfig::for_position_delete`] carries equal `file_path`
    /// bounds, so it is file-scoped and routes by path. This one is stamped with partition `x=2`, which
    /// holds no live data file, while the file it references lives in `x=1`.
    ///
    /// Two things must hold together. The read path must apply the delete, or the masked row returns on
    /// every scan. The action must not collect it, or a committed metadata change resurrects the row
    /// permanently.
    ///
    /// Mutation: revert the file-scoped leg and `removed_delete_files.is_empty()` fails. Revert the
    /// index's path map and the first `scan_y_values` assertion fails with `{10, 20, 30}`.
    #[tokio::test]
    async fn test_file_scoped_position_delete_in_a_foreign_partition_applies_and_survives() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

        // Data file A in partition x=1: rows y=10/20/30.
        let a = write_data_file(&table, "a.parquet", 1, &[
            (1, 10, 100),
            (1, 20, 200),
            (1, 30, 300),
        ])
        .await;
        let a_path = a.file_path().to_string();
        let table = append_files(&catalog, &table, vec![a]).await;

        // A file-scoped position delete for A's position 1, stamped with partition x=2, which holds no
        // live data file. A writer that stamps the table's default partitioning produces this shape.
        let delete =
            write_file_scoped_position_delete_file(&table, 2, &[(a_path.clone(), 1)]).await;
        let delete_path = delete.file_path().to_string();

        // FIXTURE SANITY — none of these may drift, or the test goes vacuously green:
        assert_eq!(
            delete.referenced_data_file(),
            None,
            "the explicit back-reference field is NOT set: this fixture exercises the BOUNDS leg, \
             through metrics the parquet writer actually computed"
        );
        assert_eq!(
            referenced_data_file_location(&delete).as_deref(),
            Some(a_path.as_str()),
            "the file_path-column bounds must pin exactly one referenced data file"
        );
        assert_eq!(
            delete.partition(),
            &Struct::from_iter([Some(Literal::long(2))]),
            "the delete is stamped with a partition tuple that differs from the data file's"
        );

        let table = add_deletes(&catalog, &table, vec![delete]).await;

        // READ PATH: the delete applies even though its partition tuple (x=2) is not the data
        // file's (x=1) — it is routed by the referenced data file's path.
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30]),
            "the file-scoped delete drops y=20 despite the partition mismatch"
        );

        // Partition x=2 holds no live data file, so the min-seq rule would call this delete dangling.
        // Removing it is an irreversible metadata change that resurrects y=20. It must be kept.
        let result = RemoveDanglingDeleteFiles::new(table.clone())
            .execute(&catalog)
            .await
            .unwrap();
        assert!(
            result.removed_delete_files.is_empty(),
            "a delete the reader still honors must never be collected, got {:?}",
            result
                .removed_delete_files
                .iter()
                .map(|file| file.file_path())
                .collect::<Vec<_>>()
        );

        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            live_delete_paths(&reloaded).await.contains(&delete_path),
            "the file-scoped delete must still be live after the action"
        );
        assert_eq!(
            scan_y_values(&reloaded).await,
            HashSet::from([10, 30]),
            "y=20 must stay masked after the maintenance action"
        );
    }

    /// The counterpart of the test above. Once the referenced data file is gone, the same file-scoped
    /// delete is genuinely dangling and is collected.
    #[tokio::test]
    async fn test_file_scoped_position_delete_is_collected_once_its_referenced_file_is_gone() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

        let a = write_data_file(&table, "a.parquet", 1, &[
            (1, 10, 100),
            (1, 20, 200),
            (1, 30, 300),
        ])
        .await;
        let a_path = a.file_path().to_string();
        let table = append_files(&catalog, &table, vec![a.clone()]).await;

        let delete =
            write_file_scoped_position_delete_file(&table, 2, &[(a_path.clone(), 1)]).await;
        let delete_path = delete.file_path().to_string();
        let table = add_deletes(&catalog, &table, vec![delete]).await;
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30]),
            "the file-scoped delete applies before the rewrite"
        );

        // RewriteFiles A -> A' (a NEW path), so the delete's reference no longer resolves.
        let a_prime = write_data_file(&table, "a-prime.parquet", 1, &[
            (1, 10, 100),
            (1, 20, 200),
            (1, 30, 300),
        ])
        .await;
        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![a], vec![a_prime]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 20, 30]),
            "after A->A' the delete references a gone file, so y=20 is already back"
        );

        let result = RemoveDanglingDeleteFiles::new(table.clone())
            .execute(&catalog)
            .await
            .unwrap();
        assert_eq!(
            result
                .removed_delete_files
                .iter()
                .map(|file| file.file_path().to_string())
                .collect::<Vec<_>>(),
            vec![delete_path.clone()],
            "the file-scoped delete whose referenced file is gone must be collected"
        );
        assert_eq!(
            result.removed_position_delete_files_count(),
            1,
            "it is counted as a parquet position delete, not a DV"
        );

        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            !live_delete_paths(&reloaded).await.contains(&delete_path),
            "the dangling file-scoped delete must be tombstoned"
        );
    }

    /// Deletion vectors, end to end. A plain RewriteFiles of data file A carries a real Puffin deletion
    /// vector forward. The vector now references a gone file, so it dangles, and this action removes it.
    /// The scan is correct before and after: A' carries no vector, so the masked row is already back.
    ///
    /// Java prunes a dangling deletion vector during RewriteFiles, so it rarely needs this action for
    /// them. The fork's RewriteFiles carries it forward and relies on this action instead.
    ///
    /// The risk pinned is a reference branch that never fires end to end.
    #[tokio::test]
    async fn test_dangling_deletion_vector_removed_after_referenced_data_rewritten_away() {
        use crate::spec::ManifestContentType;
        let (catalog, _temp) = local_fs_catalog().await;
        // V3 — deletion vectors require format version 3.
        let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

        // Data file A in partition 0: rows y=10/20/30.
        let a = write_data_file(&table, "a.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let a_path = a.file_path().to_string();
        let table = append_files(&catalog, &table, vec![a.clone()]).await;

        // A real Puffin DV referencing A, deleting position 1 (y=20). Committed via row_delta.
        let dv = write_real_dv_file(&table, "a-dv.puffin", &a_path, 0, &[1]).await;
        let dv_path = dv.file_path().to_string();
        assert!(
            is_deletion_vector(&dv),
            "fixture sanity: the written file is a PUFFIN deletion vector"
        );
        let table = add_deletes(&catalog, &table, vec![dv]).await;
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30]),
            "the DV drops y=20 before the rewrite"
        );

        // A plain RewriteFiles gives A' a new path, so the vector no longer applies and y=20 comes back.
        // The carry posture keeps the now-dangling vector.
        let a_prime = write_data_file(&table, "a-prime.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![a], vec![a_prime]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert!(
            live_delete_paths(&table).await.contains(&dv_path),
            "plain RewriteFiles carries the now-dangling DV forward (Rust carry-posture)"
        );
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 20, 30]),
            "after A->A' the DV references a gone file, so y=20 is already back"
        );

        // The action removes the dangling DV (referenced file A is gone).
        let result = RemoveDanglingDeleteFiles::new(table.clone())
            .execute(&catalog)
            .await
            .unwrap();
        assert_eq!(
            result.removed_delete_files.len(),
            1,
            "the dangling DV must be removed"
        );
        assert_eq!(result.removed_dvs_count(), 1, "it is counted as a DV");
        assert_eq!(
            result.removed_position_delete_files_count(),
            0,
            "a DV is not a parquet position delete"
        );

        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            !live_delete_paths(&reloaded).await.contains(&dv_path),
            "the dangling DV must be tombstoned"
        );
        assert_eq!(
            summary_prop(&reloaded, "removed-dvs").as_deref(),
            Some("1"),
            "the summary must report one removed DV"
        );
        // The removed DV must be a Deleted tombstone in a DELETE manifest of the new snapshot.
        let snapshot = reloaded.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(reloaded.file_io(), reloaded.metadata())
            .await
            .unwrap();
        let mut found_tombstone = false;
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != ManifestContentType::Deletes {
                continue;
            }
            let manifest = manifest_file
                .load_manifest(reloaded.file_io())
                .await
                .unwrap();
            for entry in manifest.entries() {
                if entry.file_path() == dv_path && entry.status() == ManifestStatus::Deleted {
                    found_tombstone = true;
                }
            }
        }
        assert!(
            found_tombstone,
            "the removed DV must be a Deleted tombstone"
        );

        // The read result is UNCHANGED by the GC (the DV was already not applying).
        assert_eq!(
            scan_y_values(&reloaded).await,
            HashSet::from([10, 20, 30]),
            "removing the dangling DV does not change the read result"
        );
    }
}
