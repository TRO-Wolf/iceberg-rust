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

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use futures::StreamExt;
use futures::channel::mpsc::{Sender, channel};
use tokio::sync::Notify;
use tokio::sync::futures::OwnedNotified;

use crate::runtime::spawn;
use crate::scan::{DeleteFileContext, FileScanTaskDeleteFile};
use crate::spec::{DataContentType, DataFile, DataFileFormat, Struct};
use crate::{Error, ErrorKind, Result};

/// Index of delete files
#[derive(Debug, Clone)]
pub(crate) struct DeleteFileIndex {
    state: Arc<RwLock<DeleteFileIndexState>>,
}

#[derive(Debug)]
enum DeleteFileIndexState {
    Populating(Arc<Notify>),
    Populated(PopulatedDeleteFileIndex),
    /// The populate task terminated WITHOUT publishing an index — it unwound, or the runtime was
    /// torn down and the task's future was dropped at its `collect()` await.
    ///
    /// Terminal, exactly like [`DeleteFileIndexState::Populated`]: the populate task is the sole
    /// writer and runs once, so if it dies the state can never advance on its own. Without this
    /// variant the state would stay `Populating` forever and every scan parked in
    /// [`DeleteFileIndex::get_deletes_for_data_file`] would wait on a notification that can no
    /// longer be sent. The `String` is the reason, rendered into each waiter's typed error. This
    /// mirrors `EqDelState::Failed` in [`crate::arrow::delete_filter`].
    Failed(String),
}

/// Publishes the populate task's TERMINAL state and wakes the waiters, on every exit path.
///
/// Constructed at the top of the populate task, before its first await, and disarmed by
/// [`PopulateGuard::publish`] on the success path. If the task never reaches that call — it
/// unwound (tokio drops the task's future as the panic propagates), or the runtime shut down and
/// dropped the parked future — `Drop` publishes [`DeleteFileIndexState::Failed`] instead, so every
/// waiter reaches a terminal state and gets a typed error rather than hanging forever.
///
/// Both paths write the state under the write lock and fire the notifier only AFTER releasing it,
/// so a woken waiter always observes the terminal state (see [`DeleteFileIndex::lookup_or_arm`]
/// for the other half of that handshake).
struct PopulateGuard {
    state: Arc<RwLock<DeleteFileIndexState>>,
    notify: Arc<Notify>,
    armed: bool,
}

impl PopulateGuard {
    fn new(state: Arc<RwLock<DeleteFileIndexState>>, notify: Arc<Notify>) -> Self {
        Self {
            state,
            notify,
            armed: true,
        }
    }

    /// Publish `terminal` and wake every waiter, disarming the guard.
    fn publish(&mut self, terminal: DeleteFileIndexState) {
        {
            // Recover a poisoned guard rather than cascading the panic: this is the sole writer
            // and it runs once, so recovering and completing the transition is always the right
            // move (a stranded `Populating` state would hang every waiting scan on the notifier
            // below).
            let mut guard = self
                .state
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *guard = terminal;
        }
        self.armed = false;
        self.notify.notify_waiters();
    }
}

impl Drop for PopulateGuard {
    fn drop(&mut self) {
        if self.armed {
            self.publish(DeleteFileIndexState::Failed(
                "the delete file index populate task terminated before publishing an index \
                 (it panicked, or the runtime was shut down)"
                    .to_string(),
            ));
        }
    }
}

/// The outcome of consulting the index state once — see [`DeleteFileIndex::lookup_or_arm`].
enum IndexLookup {
    /// The index had reached a terminal state; this is the answer.
    Ready(Vec<FileScanTaskDeleteFile>),
    /// The index was still populating. Await this future, then read the state again.
    Wait(OwnedNotified),
}

#[derive(Debug)]
struct PopulatedDeleteFileIndex {
    #[allow(dead_code)]
    global_equality_deletes: Vec<Arc<DeleteFileContext>>,
    eq_deletes_by_partition: HashMap<Struct, Vec<Arc<DeleteFileContext>>>,
    pos_deletes_by_partition: HashMap<Struct, Vec<Arc<DeleteFileContext>>>,
    // TODO: do we need this?
    // pos_deletes_by_path: HashMap<String, Vec<Arc<DeleteFileContext>>>,
    /// Deletion vectors keyed by the data file they apply to (the DV's
    /// `referenced_data_file`), mirroring Java `DeleteFileIndex.dvByPath`
    /// (`DeleteFileIndex.Builder.build` L500/L505-506: a POSITION_DELETES file with
    /// `ContentFileUtil.isDV` — format == PUFFIN — is indexed by `referencedDataFile()`).
    ///
    /// A valid table has AT MOST ONE DV per data file; Java's `add(dvByPath, dv)` (L528-535)
    /// raises `ValidationException` ("Can't index multiple DVs for %s") on a duplicate. This
    /// index's lookup signature is infallible, so duplicates are kept HERE and rejected
    /// fail-loud at the load door instead (`CachingDeleteFileLoader::load_deletes`).
    dv_by_path: HashMap<String, Vec<Arc<DeleteFileContext>>>,
}

/// Whether a delete file is a deletion vector. Java `ContentFileUtil.isDV` (L142-144):
/// `deleteFile.format() == FileFormat.PUFFIN`.
pub(crate) fn is_deletion_vector(data_file: &DataFile) -> bool {
    data_file.file_format() == DataFileFormat::Puffin
}

impl DeleteFileIndex {
    /// create a new `DeleteFileIndex` along with the sender that populates it with delete files
    pub(crate) fn new() -> (DeleteFileIndex, Sender<DeleteFileContext>) {
        // TODO: what should the channel limit be?
        let (tx, rx) = channel(10);
        let notify = Arc::new(Notify::new());
        let state = Arc::new(RwLock::new(DeleteFileIndexState::Populating(
            notify.clone(),
        )));
        let delete_file_stream = rx.boxed();

        spawn({
            let state = state.clone();
            async move {
                // Armed BEFORE the first await, so every way out of this task — including the
                // runtime dropping the future while it is parked on `collect()` below — publishes
                // a terminal state and wakes the waiters.
                let mut guard = PopulateGuard::new(state, notify);

                let delete_files: Vec<DeleteFileContext> =
                    delete_file_stream.collect::<Vec<_>>().await;

                let populated_delete_file_index = PopulatedDeleteFileIndex::new(delete_files);

                guard.publish(DeleteFileIndexState::Populated(populated_delete_file_index));
            }
        });

        (DeleteFileIndex { state }, tx)
    }

    /// Gets all the delete files that apply to the specified data file.
    ///
    /// Fallible: a deletion vector whose data sequence number is LESS THAN the data file's marks an
    /// invalid table and returns an [`ErrorKind::DataInvalid`] error (Java
    /// `DeleteFileIndex.findDV` `ValidationException`) — see
    /// [`PopulatedDeleteFileIndex::get_deletes_for_data_file`].
    pub(crate) async fn get_deletes_for_data_file(
        &self,
        data_file: &DataFile,
        seq_num: Option<i64>,
    ) -> Result<Vec<FileScanTaskDeleteFile>> {
        match self.lookup_or_arm(data_file, seq_num)? {
            IndexLookup::Ready(deletes) => return Ok(deletes),
            IndexLookup::Wait(notified) => notified.await,
        }

        // The populate task publishes a TERMINAL state under the write lock and only then fires
        // the notifier, so a woken waiter always observes `Populated` or `Failed`. (This second
        // call arms a notifier it will not use — one `Arc` clone — which keeps the lock/arm
        // handshake in exactly one place.)
        match self.lookup_or_arm(data_file, seq_num)? {
            IndexLookup::Ready(deletes) => Ok(deletes),
            IndexLookup::Wait(_) => Err(Error::new(
                ErrorKind::Unexpected,
                "delete file index notifier fired but the index is not populated",
            )),
        }
    }

    /// Read the index state once: answer outright if it is terminal, otherwise ARM the notifier.
    ///
    /// The arming MUST happen here, while the read lock is still held.
    /// [`Notify::notify_waiters`] stores no permit and only wakes `Notified` futures that already
    /// EXIST when it fires, and `Notify::notified_owned` snapshots the notifier's
    /// `notify_waiters` counter at CALL time — so a `Notified` created after the populate task
    /// fired is never woken. Returning a bare `Arc<Notify>` and calling `.notified()` at the await
    /// site left exactly that window open between releasing the read lock and creating the future:
    /// if the populate task published and notified in it, the wakeup was dropped and the scan
    /// awaited forever (upstream apache/iceberg-rust#2696, the same class as #2859 on the
    /// positional-delete wait path).
    ///
    /// Creating the future under the read lock closes the window: the populate task cannot fire
    /// the notifier until it has taken the WRITE lock, which cannot be granted while this read
    /// lock is held, so any notification necessarily follows this arming and is delivered.
    fn lookup_or_arm(&self, data_file: &DataFile, seq_num: Option<i64>) -> Result<IndexLookup> {
        let guard = self.state.read().map_err(|_| {
            Error::new(
                ErrorKind::Unexpected,
                "delete file index RwLock was poisoned",
            )
        })?;

        match &*guard {
            DeleteFileIndexState::Populated(index) => Ok(IndexLookup::Ready(
                index.get_deletes_for_data_file(data_file, seq_num)?,
            )),
            DeleteFileIndexState::Failed(reason) => Err(Error::new(
                ErrorKind::Unexpected,
                format!("delete file index is unavailable: {reason}"),
            )),
            DeleteFileIndexState::Populating(notifier) => {
                Ok(IndexLookup::Wait(notifier.clone().notified_owned()))
            }
        }
    }
}

impl PopulatedDeleteFileIndex {
    /// Creates a new populated delete file index from a list of delete file contexts, which
    /// allows for fast lookup when determining which delete files apply to a given data file.
    ///
    /// 1. The partition information is extracted from each delete file's manifest entry.
    /// 2. If the partition is empty and the delete file is not a positional delete,
    ///    it is added to the `global_equality_deletes` vector
    /// 3. Otherwise, the delete file is added to one of two hash maps based on its content type.
    fn new(files: Vec<DeleteFileContext>) -> PopulatedDeleteFileIndex {
        let mut eq_deletes_by_partition: HashMap<Struct, Vec<Arc<DeleteFileContext>>> =
            HashMap::default();
        let mut pos_deletes_by_partition: HashMap<Struct, Vec<Arc<DeleteFileContext>>> =
            HashMap::default();
        let mut dv_by_path: HashMap<String, Vec<Arc<DeleteFileContext>>> = HashMap::default();

        let mut global_equality_deletes: Vec<Arc<DeleteFileContext>> = vec![];

        files.into_iter().for_each(|ctx| {
            let arc_ctx = Arc::new(ctx);

            // A deletion vector is FILE-scoped: it indexes by the data file it references, never
            // by partition (Java `DeleteFileIndex.Builder.build` L505-506 routes
            // POSITION_DELETES + `isDV` to `dvByPath` keyed by `referencedDataFile()`). A Puffin
            // position delete WITHOUT a referenced data file is invalid per the Puffin spec
            // (`referenced-data-file` is mandatory for `deletion-vector-v1`); it falls through to
            // the partition map so the loader's DV dispatch rejects it loudly by name instead of
            // it being silently dropped here.
            if arc_ctx.manifest_entry.content_type() == DataContentType::PositionDeletes
                && is_deletion_vector(arc_ctx.manifest_entry.data_file())
                && let Some(referenced_data_file) =
                    arc_ctx.manifest_entry.data_file().referenced_data_file()
            {
                dv_by_path
                    .entry(referenced_data_file)
                    .or_default()
                    .push(arc_ctx);
                return;
            }

            let partition = arc_ctx.manifest_entry.data_file().partition();

            // The spec states that "Equality delete files stored with an unpartitioned spec are applied as global deletes".
            if partition.fields().is_empty() {
                // TODO: confirm we're good to skip here if we encounter a pos del
                if arc_ctx.manifest_entry.content_type() != DataContentType::PositionDeletes {
                    global_equality_deletes.push(arc_ctx);
                    return;
                }
            }

            let destination_map = match arc_ctx.manifest_entry.content_type() {
                DataContentType::PositionDeletes => &mut pos_deletes_by_partition,
                DataContentType::EqualityDeletes => &mut eq_deletes_by_partition,
                // A `Data`-typed entry cannot legitimately reach the delete-file index:
                // `TableScan::process_delete_manifest_entry` (scan/mod.rs) rejects any data-file
                // entry found in a delete manifest before it is ever sent here. Skip it defensively
                // rather than panicking the populate task — a panic there strands every waiting
                // scan on the populate notifier, which would then never fire. Matching `Data`
                // explicitly (instead of a `_` arm) also turns a future `DataContentType` variant
                // into a compile error at this site, forcing a routing decision rather than a
                // silent insert into the wrong map.
                DataContentType::Data => return,
            };

            destination_map
                .entry(partition.clone())
                .and_modify(|entry| {
                    entry.push(arc_ctx.clone());
                })
                .or_insert(vec![arc_ctx.clone()]);
        });

        PopulatedDeleteFileIndex {
            global_equality_deletes,
            eq_deletes_by_partition,
            pos_deletes_by_partition,
            dv_by_path,
        }
    }

    /// Determine all the delete files that apply to the provided `DataFile`.
    ///
    /// FALLIBLE because of the deletion-vector sequence-number validation (Java
    /// `DeleteFileIndex.findDV` L208-214, 1.10.0-bytecode-verified): a DV attached to a data file
    /// MUST have `dv.dataSequenceNumber() >= dataFile.dataSequenceNumber()`, or the table is
    /// invalid and the scan must fail loud rather than silently apply the wrong DV. The lookup was
    /// infallible before this validation landed (D1's deferred residue); the ripple was assessed
    /// small (one production caller — `scan/context.rs`, already `Result`-returning) and the index
    /// is the ONLY place both sequence numbers are in hand (`seq_num` = the data file's, the DV's via
    /// its manifest entry), so the check lives here rather than at the load door (the caching loader
    /// never receives either sequence number — `FileScanTaskDeleteFile` drops them).
    fn get_deletes_for_data_file(
        &self,
        data_file: &DataFile,
        seq_num: Option<i64>,
    ) -> Result<Vec<FileScanTaskDeleteFile>> {
        let mut results = vec![];

        self.global_equality_deletes
            .iter()
            // filter that returns true if the provided delete file's sequence number is **greater than** `seq_num`
            .filter(|&delete| {
                seq_num
                    .map(|seq_num| delete.manifest_entry.sequence_number() > Some(seq_num))
                    .unwrap_or_else(|| true)
            })
            .for_each(|delete| results.push(delete.as_ref().into()));

        if let Some(deletes) = self.eq_deletes_by_partition.get(data_file.partition()) {
            deletes
                .iter()
                // filter that returns true if the provided delete file's sequence number is **greater than** `seq_num`
                .filter(|&delete| {
                    seq_num
                        .map(|seq_num| delete.manifest_entry.sequence_number() > Some(seq_num))
                        .unwrap_or_else(|| true)
                        && data_file.partition_spec_id == delete.partition_spec_id
                })
                .for_each(|delete| results.push(delete.as_ref().into()));
        }

        // A data file with a DELETION VECTOR uses the DV INSTEAD of any parquet position
        // deletes: Java `DeleteFileIndex.forDataFile` (L156-167) returns
        // {global eq, partition eq, dv} when `findDV` hits and only consults the
        // position-delete maps when it does not. The DV lookup is by the data file's PATH
        // (Java `findDV` L202-216: `dvByPath.get(dataFile.location())`) and is NOT
        // sequence-filtered. Instead Java VALIDATES `dv.dataSequenceNumber() >= seq` (the data
        // file's sequence number) and throws a `ValidationException` otherwise (L208-214,
        // 1.10.0-bytecode-verified) — a DV must never be attached to a data file from a LATER
        // sequence number. That validation now lives HERE (was D1's deferred residue): the index
        // is the only place both sequence numbers are in hand. Duplicate DVs for one path (Java's
        // other ValidationException, L528-535) are all returned so the loader rejects them loudly.
        if let Some(dvs) = self.dv_by_path.get(data_file.file_path()) {
            for delete in dvs {
                // Java `findDV` L208-214: a DV's data sequence number must be >= the data file's.
                // `seq_num` is the data file's data sequence number (Java `seq`); the DV's is its
                // manifest entry's. A `None` data-file seq (only in unit fixtures that pass
                // `seq_num = None`) cannot be violated, so it is treated as valid — Java always has
                // a concrete `seq`. The mirror is the EXACT 1.10.0 message.
                if let Some(data_seq) = seq_num
                    && let Some(dv_seq) = delete.manifest_entry.sequence_number()
                    && dv_seq < data_seq
                {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "DV data sequence number ({dv_seq}) must be greater than or equal to data file sequence number ({data_seq})"
                        ),
                    ));
                }
                results.push(delete.as_ref().into());
            }
            return Ok(results);
        }

        // TODO: the spec states that:
        //     "The data file's file_path is equal to the delete file's referenced_data_file if it is non-null".
        //     we're not yet doing that here. The referenced data file's name will also be present in the positional
        //     delete file's file path column.
        if let Some(deletes) = self.pos_deletes_by_partition.get(data_file.partition()) {
            deletes
                .iter()
                // filter that returns true if the provided delete file's sequence number is **greater than or equal to** `seq_num`
                .filter(|&delete| {
                    seq_num
                        .map(|seq_num| delete.manifest_entry.sequence_number() >= Some(seq_num))
                        .unwrap_or_else(|| true)
                        && data_file.partition_spec_id == delete.partition_spec_id
                })
                .for_each(|delete| results.push(delete.as_ref().into()));
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use uuid::Uuid;

    use super::*;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestEntry, ManifestStatus,
        Struct,
    };

    #[test]
    fn test_delete_file_index_unpartitioned() {
        let deletes: Vec<ManifestEntry> = vec![
            build_added_manifest_entry(4, &build_unpartitioned_eq_delete()),
            build_added_manifest_entry(6, &build_unpartitioned_eq_delete()),
            build_added_manifest_entry(5, &build_unpartitioned_pos_delete()),
            build_added_manifest_entry(6, &build_unpartitioned_pos_delete()),
        ];

        let delete_file_paths: Vec<String> = deletes
            .iter()
            .map(|file| file.file_path().to_string())
            .collect();

        let delete_contexts: Vec<DeleteFileContext> = deletes
            .into_iter()
            .map(|entry| DeleteFileContext {
                manifest_entry: entry.into(),
                partition_spec_id: 0,
            })
            .collect();

        let delete_file_index = PopulatedDeleteFileIndex::new(delete_contexts);

        let data_file = build_unpartitioned_data_file();

        // All deletes apply to sequence 0
        let delete_files_to_apply_for_seq_0 = delete_file_index
            .get_deletes_for_data_file(&data_file, Some(0))
            .unwrap();
        assert_eq!(delete_files_to_apply_for_seq_0.len(), 4);

        // All deletes apply to sequence 3
        let delete_files_to_apply_for_seq_3 = delete_file_index
            .get_deletes_for_data_file(&data_file, Some(3))
            .unwrap();
        assert_eq!(delete_files_to_apply_for_seq_3.len(), 4);

        // Last 3 deletes apply to sequence 4
        let delete_files_to_apply_for_seq_4 = delete_file_index
            .get_deletes_for_data_file(&data_file, Some(4))
            .unwrap();
        let actual_paths_to_apply_for_seq_4: Vec<String> = delete_files_to_apply_for_seq_4
            .into_iter()
            .map(|file| file.file_path)
            .collect();

        assert_eq!(
            actual_paths_to_apply_for_seq_4,
            delete_file_paths[delete_file_paths.len() - 3..]
        );

        // Last 3 deletes apply to sequence 5
        let delete_files_to_apply_for_seq_5 = delete_file_index
            .get_deletes_for_data_file(&data_file, Some(5))
            .unwrap();
        let actual_paths_to_apply_for_seq_5: Vec<String> = delete_files_to_apply_for_seq_5
            .into_iter()
            .map(|file| file.file_path)
            .collect();
        assert_eq!(
            actual_paths_to_apply_for_seq_5,
            delete_file_paths[delete_file_paths.len() - 3..]
        );

        // Only the last position delete applies to sequence 6
        let delete_files_to_apply_for_seq_6 = delete_file_index
            .get_deletes_for_data_file(&data_file, Some(6))
            .unwrap();
        let actual_paths_to_apply_for_seq_6: Vec<String> = delete_files_to_apply_for_seq_6
            .into_iter()
            .map(|file| file.file_path)
            .collect();
        assert_eq!(
            actual_paths_to_apply_for_seq_6,
            delete_file_paths[delete_file_paths.len() - 1..]
        );

        // The 2 global equality deletes should match against any partitioned file
        let partitioned_file =
            build_partitioned_data_file(&Struct::from_iter([Some(Literal::long(100))]), 1);

        let delete_files_to_apply_for_partitioned_file = delete_file_index
            .get_deletes_for_data_file(&partitioned_file, Some(0))
            .unwrap();
        let actual_paths_to_apply_for_partitioned_file: Vec<String> =
            delete_files_to_apply_for_partitioned_file
                .into_iter()
                .map(|file| file.file_path)
                .collect();
        assert_eq!(
            actual_paths_to_apply_for_partitioned_file,
            delete_file_paths[..2]
        );
    }

    #[test]
    fn test_delete_file_index_partitioned() {
        let partition_one = Struct::from_iter([Some(Literal::long(100))]);
        let spec_id = 1;
        let deletes: Vec<ManifestEntry> = vec![
            build_added_manifest_entry(4, &build_partitioned_eq_delete(&partition_one, spec_id)),
            build_added_manifest_entry(6, &build_partitioned_eq_delete(&partition_one, spec_id)),
            build_added_manifest_entry(5, &build_partitioned_pos_delete(&partition_one, spec_id)),
            build_added_manifest_entry(6, &build_partitioned_pos_delete(&partition_one, spec_id)),
        ];

        let delete_file_paths: Vec<String> = deletes
            .iter()
            .map(|file| file.file_path().to_string())
            .collect();

        let delete_contexts: Vec<DeleteFileContext> = deletes
            .into_iter()
            .map(|entry| DeleteFileContext {
                manifest_entry: entry.into(),
                partition_spec_id: spec_id,
            })
            .collect();

        let delete_file_index = PopulatedDeleteFileIndex::new(delete_contexts);

        let partitioned_file =
            build_partitioned_data_file(&Struct::from_iter([Some(Literal::long(100))]), spec_id);

        // All deletes apply to sequence 0
        let delete_files_to_apply_for_seq_0 = delete_file_index
            .get_deletes_for_data_file(&partitioned_file, Some(0))
            .unwrap();
        assert_eq!(delete_files_to_apply_for_seq_0.len(), 4);

        // All deletes apply to sequence 3
        let delete_files_to_apply_for_seq_3 = delete_file_index
            .get_deletes_for_data_file(&partitioned_file, Some(3))
            .unwrap();
        assert_eq!(delete_files_to_apply_for_seq_3.len(), 4);

        // Last 3 deletes apply to sequence 4
        let delete_files_to_apply_for_seq_4 = delete_file_index
            .get_deletes_for_data_file(&partitioned_file, Some(4))
            .unwrap();
        let actual_paths_to_apply_for_seq_4: Vec<String> = delete_files_to_apply_for_seq_4
            .into_iter()
            .map(|file| file.file_path)
            .collect();

        assert_eq!(
            actual_paths_to_apply_for_seq_4,
            delete_file_paths[delete_file_paths.len() - 3..]
        );

        // Last 3 deletes apply to sequence 5
        let delete_files_to_apply_for_seq_5 = delete_file_index
            .get_deletes_for_data_file(&partitioned_file, Some(5))
            .unwrap();
        let actual_paths_to_apply_for_seq_5: Vec<String> = delete_files_to_apply_for_seq_5
            .into_iter()
            .map(|file| file.file_path)
            .collect();
        assert_eq!(
            actual_paths_to_apply_for_seq_5,
            delete_file_paths[delete_file_paths.len() - 3..]
        );

        // Only the last position delete applies to sequence 6
        let delete_files_to_apply_for_seq_6 = delete_file_index
            .get_deletes_for_data_file(&partitioned_file, Some(6))
            .unwrap();
        let actual_paths_to_apply_for_seq_6: Vec<String> = delete_files_to_apply_for_seq_6
            .into_iter()
            .map(|file| file.file_path)
            .collect();
        assert_eq!(
            actual_paths_to_apply_for_seq_6,
            delete_file_paths[delete_file_paths.len() - 1..]
        );

        // Data file with different partition tuples does not match any delete files
        let partitioned_second_file =
            build_partitioned_data_file(&Struct::from_iter([Some(Literal::long(200))]), 1);
        let delete_files_to_apply_for_different_partition = delete_file_index
            .get_deletes_for_data_file(&partitioned_second_file, Some(0))
            .unwrap();
        let actual_paths_to_apply_for_different_partition: Vec<String> =
            delete_files_to_apply_for_different_partition
                .into_iter()
                .map(|file| file.file_path)
                .collect();
        assert!(actual_paths_to_apply_for_different_partition.is_empty());

        // Data file with same tuple but different spec ID does not match any delete files
        let partitioned_different_spec = build_partitioned_data_file(&partition_one, 2);
        let delete_files_to_apply_for_different_spec = delete_file_index
            .get_deletes_for_data_file(&partitioned_different_spec, Some(0))
            .unwrap();
        let actual_paths_to_apply_for_different_spec: Vec<String> =
            delete_files_to_apply_for_different_spec
                .into_iter()
                .map(|file| file.file_path)
                .collect();
        assert!(actual_paths_to_apply_for_different_spec.is_empty());
    }

    fn build_unpartitioned_eq_delete() -> DataFile {
        build_partitioned_eq_delete(&Struct::empty(), 0)
    }

    fn build_partitioned_eq_delete(partition: &Struct, spec_id: i32) -> DataFile {
        DataFileBuilder::default()
            .file_path(format!("{}_equality_delete.parquet", Uuid::new_v4()))
            .file_format(DataFileFormat::Parquet)
            .content(DataContentType::EqualityDeletes)
            .equality_ids(Some(vec![1]))
            .record_count(1)
            .partition(partition.clone())
            .partition_spec_id(spec_id)
            .file_size_in_bytes(100)
            .build()
            .unwrap()
    }

    fn build_unpartitioned_pos_delete() -> DataFile {
        build_partitioned_pos_delete(&Struct::empty(), 0)
    }

    fn build_partitioned_pos_delete(partition: &Struct, spec_id: i32) -> DataFile {
        DataFileBuilder::default()
            .file_path(format!("{}-pos-delete.parquet", Uuid::new_v4()))
            .file_format(DataFileFormat::Parquet)
            .content(DataContentType::PositionDeletes)
            .record_count(1)
            .referenced_data_file(Some("/some-data-file.parquet".to_string()))
            .partition(partition.clone())
            .partition_spec_id(spec_id)
            .file_size_in_bytes(100)
            .build()
            .unwrap()
    }

    /// Build a deletion vector `DataFile`: a POSITION_DELETES entry in PUFFIN format referencing
    /// `referenced_data_file`, with blob coordinates (the discriminator Java uses is the format
    /// — `ContentFileUtil.isDV`).
    fn build_partitioned_deletion_vector(
        referenced_data_file: &str,
        partition: &Struct,
        spec_id: i32,
    ) -> DataFile {
        DataFileBuilder::default()
            .file_path(format!("{}-deletes.puffin", Uuid::new_v4()))
            .file_format(DataFileFormat::Puffin)
            .content(DataContentType::PositionDeletes)
            .record_count(2)
            .referenced_data_file(Some(referenced_data_file.to_string()))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40))
            .partition(partition.clone())
            .partition_spec_id(spec_id)
            .file_size_in_bytes(100)
            .build()
            .unwrap()
    }

    /// Risk pinned: Java `DeleteFileIndex.forDataFile` (L156-167) — a data file with a DV gets
    /// the DV INSTEAD of any parquet position deletes (the DV is the complete position-delete
    /// state for that file; also returning the parquet deletes would re-apply superseded
    /// deletes), while equality deletes still apply alongside it.
    #[test]
    fn test_dv_supersedes_position_deletes_and_keeps_equality_deletes() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let spec_id = 1;
        let data_file = build_partitioned_data_file(&partition, spec_id);

        let deletes: Vec<ManifestEntry> = vec![
            build_added_manifest_entry(2, &build_partitioned_eq_delete(&partition, spec_id)),
            build_added_manifest_entry(2, &build_partitioned_pos_delete(&partition, spec_id)),
            build_added_manifest_entry(
                2,
                &build_partitioned_deletion_vector(data_file.file_path(), &partition, spec_id),
            ),
        ];
        let eq_delete_path = deletes[0].file_path().to_string();
        let dv_path = deletes[2].file_path().to_string();

        let delete_contexts: Vec<DeleteFileContext> = deletes
            .into_iter()
            .map(|entry| DeleteFileContext {
                manifest_entry: entry.into(),
                partition_spec_id: spec_id,
            })
            .collect();
        let index = PopulatedDeleteFileIndex::new(delete_contexts);

        let results = index
            .get_deletes_for_data_file(&data_file, Some(1))
            .unwrap();
        let result_paths: Vec<&str> = results.iter().map(|f| f.file_path.as_str()).collect();

        assert_eq!(
            result_paths,
            vec![eq_delete_path.as_str(), dv_path.as_str()],
            "expected the equality delete + the DV, and NO parquet position delete"
        );
        assert_eq!(
            results[1].file_format,
            DataFileFormat::Puffin,
            "the DV entry must carry the Puffin format discriminator to the loader"
        );
        assert_eq!(
            results[1].referenced_data_file.as_deref(),
            Some(data_file.file_path()),
            "the DV entry must carry the referenced data file for keying"
        );
    }

    /// Risk pinned: a DV is FILE-scoped (keyed by `referenced_data_file`, Java `findDV`
    /// L202-216) — a SIBLING data file in the SAME partition must NOT receive it, and still
    /// receives the partition-scoped parquet position deletes.
    #[test]
    fn test_dv_does_not_apply_to_sibling_file_in_same_partition() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let spec_id = 1;
        let data_file_with_dv = build_partitioned_data_file(&partition, spec_id);
        let sibling_data_file = build_partitioned_data_file(&partition, spec_id);

        let deletes: Vec<ManifestEntry> = vec![
            build_added_manifest_entry(2, &build_partitioned_pos_delete(&partition, spec_id)),
            build_added_manifest_entry(
                2,
                &build_partitioned_deletion_vector(
                    data_file_with_dv.file_path(),
                    &partition,
                    spec_id,
                ),
            ),
        ];
        let pos_delete_path = deletes[0].file_path().to_string();
        let dv_path = deletes[1].file_path().to_string();

        let delete_contexts: Vec<DeleteFileContext> = deletes
            .into_iter()
            .map(|entry| DeleteFileContext {
                manifest_entry: entry.into(),
                partition_spec_id: spec_id,
            })
            .collect();
        let index = PopulatedDeleteFileIndex::new(delete_contexts);

        let sibling_results = index
            .get_deletes_for_data_file(&sibling_data_file, Some(1))
            .unwrap();
        let sibling_paths: Vec<&str> = sibling_results
            .iter()
            .map(|f| f.file_path.as_str())
            .collect();
        assert_eq!(
            sibling_paths,
            vec![pos_delete_path.as_str()],
            "the sibling file gets the partition-scoped parquet position delete, never the DV"
        );

        let dv_file_results = index
            .get_deletes_for_data_file(&data_file_with_dv, Some(1))
            .unwrap();
        let dv_file_paths: Vec<&str> = dv_file_results
            .iter()
            .map(|f| f.file_path.as_str())
            .collect();
        assert_eq!(dv_file_paths, vec![dv_path.as_str()]);
    }

    /// Risk pinned: a DV is NOT seq-FILTERED (Java `findDV` does not drop it by sequence number —
    /// dropping a valid DV would resurrect deleted rows), and the VALID boundary is returned. With
    /// a DV at data seq 5: `dv_seq == data_seq` (5 == 5, the row-delta same-snapshot-family case)
    /// and `dv_seq > data_seq` (5 > 3) both return the DV. The INVALID case (`dv_seq < data_seq`)
    /// is the separate `test_dv_lower_seq_than_data_file_is_invalid_table` test.
    #[test]
    fn test_dv_is_not_sequence_filtered_at_valid_boundary() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let spec_id = 1;
        let data_file = build_partitioned_data_file(&partition, spec_id);

        let dv_entry = build_added_manifest_entry(
            5,
            &build_partitioned_deletion_vector(data_file.file_path(), &partition, spec_id),
        );
        let index = PopulatedDeleteFileIndex::new(vec![DeleteFileContext {
            manifest_entry: dv_entry.into(),
            partition_spec_id: spec_id,
        }]);

        // dv_seq == data_seq (5 == 5) — the row-delta case; the DV applies.
        assert_eq!(
            index
                .get_deletes_for_data_file(&data_file, Some(5))
                .unwrap()
                .len(),
            1
        );
        // dv_seq > data_seq (5 > 3) — a later DV on earlier data; the DV applies.
        assert_eq!(
            index
                .get_deletes_for_data_file(&data_file, Some(3))
                .unwrap()
                .len(),
            1
        );
    }

    /// Risk pinned (the dv_seq residue, now landed): a DV whose data sequence number is LESS THAN
    /// the data file's marks an INVALID table — Java `DeleteFileIndex.findDV` throws a
    /// `ValidationException` (L208-214, 1.10.0-bytecode-verified). A valid writer never produces
    /// this, so the metadata is hand-built: a DV at data seq 5 looked up against a data file at
    /// seq 9 (5 < 9). The lookup must fail LOUD with the EXACT Java message naming both sequence
    /// numbers, never silently apply the wrong DV.
    ///
    /// MUTATION: disabling the check (returning the DV regardless) makes this test see a silent
    /// `Ok(vec![dv])` instead of the error.
    #[test]
    fn test_dv_lower_seq_than_data_file_is_invalid_table() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let spec_id = 1;
        let data_file = build_partitioned_data_file(&partition, spec_id);

        // Hand-built invalid metadata: a DV committed at data seq 5, looked up for a data file at
        // seq 9 — `dv_seq (5) < data_seq (9)`, which no valid writer produces.
        let dv_entry = build_added_manifest_entry(
            5,
            &build_partitioned_deletion_vector(data_file.file_path(), &partition, spec_id),
        );
        let index = PopulatedDeleteFileIndex::new(vec![DeleteFileContext {
            manifest_entry: dv_entry.into(),
            partition_spec_id: spec_id,
        }]);

        let err = index
            .get_deletes_for_data_file(&data_file, Some(9))
            .expect_err("a DV from an earlier sequence number than the data file is invalid");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            err.message(),
            "DV data sequence number (5) must be greater than or equal to data file sequence number (9)",
            "the message must mirror Java DeleteFileIndex.findDV exactly (1.10.0 bytecode)"
        );
    }

    /// Risk pinned: a PUFFIN position delete WITHOUT `referenced_data_file` is invalid (the
    /// Puffin spec makes the property mandatory for DVs). It must NOT be silently dropped by the
    /// index — it falls through to the partition map so the loader rejects it by name.
    #[test]
    fn test_puffin_delete_without_referenced_file_reaches_loader_for_rejection() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let spec_id = 1;
        let data_file = build_partitioned_data_file(&partition, spec_id);

        let invalid_dv = DataFileBuilder::default()
            .file_path("orphan-deletes.puffin".to_string())
            .file_format(DataFileFormat::Puffin)
            .content(DataContentType::PositionDeletes)
            .record_count(2)
            .partition(partition.clone())
            .partition_spec_id(spec_id)
            .file_size_in_bytes(100)
            .build()
            .unwrap();

        let index = PopulatedDeleteFileIndex::new(vec![DeleteFileContext {
            manifest_entry: build_added_manifest_entry(2, &invalid_dv).into(),
            partition_spec_id: spec_id,
        }]);

        let results = index
            .get_deletes_for_data_file(&data_file, Some(1))
            .unwrap();
        assert_eq!(results.len(), 1, "the invalid DV must reach the loader");
        assert_eq!(results[0].file_format, DataFileFormat::Puffin);
        assert_eq!(results[0].referenced_data_file, None);
    }

    /// Risk pinned (audit SAF-003): a poisoned index `state` lock must surface a typed error from
    /// `get_deletes_for_data_file`, never a panic. The sender is kept alive so the populate task
    /// stays parked (`Populating`) and never touches the lock — the poison below is the only thing
    /// that can fail the read. MUTATION: restoring `self.state.read().unwrap()` turns the read on
    /// the poisoned lock into a panic that propagates through the awaited call (RED).
    #[tokio::test]
    async fn test_poisoned_index_state_yields_typed_error_not_panic() {
        let (index, _tx) = DeleteFileIndex::new();

        let poisoner = index.clone();
        let handle = std::thread::spawn(move || {
            let _guard = poisoner
                .state
                .write()
                .expect("acquire write guard to poison");
            panic!("intentionally poison the delete-file-index state lock");
        });
        assert!(
            handle.join().is_err(),
            "the poisoning thread must have panicked while holding the guard"
        );

        let data_file = build_unpartitioned_data_file();
        let error = index
            .get_deletes_for_data_file(&data_file, Some(0))
            .await
            .expect_err("a poisoned index lock must surface a typed error, not panic");
        assert_eq!(error.kind(), ErrorKind::Unexpected);
    }

    /// Risk pinned (audit SAF-003, `:185` unreachable): a `Data`-typed delete context cannot occur
    /// in production (`process_delete_manifest_entry` rejects data-file entries in a delete
    /// manifest), but if one ever reaches the index builder it must be SKIPPED defensively, never
    /// panic the populate task. A non-empty partition routes it to the content-type match (an empty
    /// partition would divert it to `global_equality_deletes` first). MUTATION: restoring
    /// `_ => unreachable!()` panics this test.
    #[test]
    fn test_data_typed_delete_context_is_skipped_not_panicked() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let spec_id = 1;

        let data_ctx = DeleteFileContext {
            manifest_entry: build_added_manifest_entry(
                2,
                &build_partitioned_data_file(&partition, spec_id),
            )
            .into(),
            partition_spec_id: spec_id,
        };

        let index = PopulatedDeleteFileIndex::new(vec![data_ctx]);

        let data_file = build_partitioned_data_file(&partition, spec_id);
        let results = index
            .get_deletes_for_data_file(&data_file, Some(0))
            .expect("skipping a Data-typed entry must not error");
        assert!(
            results.is_empty(),
            "a Data-typed entry must be skipped, not indexed as a delete"
        );
    }

    /// Risk pinned (audit SAF-007 / upstream apache/iceberg-rust#2696): the waiter must ARM its
    /// notifier while it still holds the read lock, so a `notify_waiters()` that fires before the
    /// waiter awaits still wakes it. `notify_waiters()` stores no permit, so this test's ordering
    /// — publish FIRST, await SECOND — only completes when the `Notified` already existed at
    /// publish time.
    ///
    /// MUTATION (semantic revert to the base contract: `IndexLookup::Wait` carries a raw
    /// `Arc<Notify>` and the waiter calls `.notified()` at the await site): the future is created
    /// after `publish`, the wakeup is lost, and the timeout below fires (RED).
    #[tokio::test]
    async fn test_waiter_is_armed_before_the_publisher_can_notify() {
        // The sender stays alive for the whole test, so the real populate task stays parked on
        // its `collect()` and this test is the only publisher.
        let (index, _tx) = DeleteFileIndex::new();
        let data_file = build_unpartitioned_data_file();

        let notified = match index
            .lookup_or_arm(&data_file, Some(0))
            .expect("arming a populating index must not error")
        {
            IndexLookup::Wait(notified) => notified,
            IndexLookup::Ready(_) => {
                panic!("the index must still be populating while the sender is alive")
            }
        };

        // Publish + `notify_waiters()` through the production publisher, BEFORE the waiter awaits.
        let notifier = {
            let guard = index.state.read().expect("read the index state");
            match &*guard {
                DeleteFileIndexState::Populating(notifier) => notifier.clone(),
                other => panic!("expected a populating index, got {other:?}"),
            }
        };
        PopulateGuard::new(index.state.clone(), notifier).publish(DeleteFileIndexState::Populated(
            PopulatedDeleteFileIndex::new(vec![]),
        ));

        tokio::time::timeout(Duration::from_secs(5), notified)
            .await
            .expect("a notification fired after arming must wake the waiter, not be lost");

        let deletes = index
            .get_deletes_for_data_file(&data_file, Some(0))
            .await
            .expect("the woken waiter must read the published index");
        assert!(
            deletes.is_empty(),
            "the published index was empty, so no deletes may apply"
        );
    }

    /// Risk pinned (audit SAF-007): if the populate task dies WITHOUT publishing — here by tearing
    /// down the runtime that hosts it while it is parked on `collect()` — the index must reach the
    /// terminal `Failed` state so every waiter gets a typed error. Nothing can advance the state
    /// afterwards, so without that transition the waiter parks on a notification that can never be
    /// sent.
    ///
    /// MUTATION: reverting `delete_file_index.rs` (no `PopulateGuard`, no `Failed`) leaves the
    /// state at `Populating`; the waiter never wakes and the timeout below fires (RED).
    #[tokio::test]
    async fn test_dead_populate_task_yields_a_typed_error_not_a_hang() {
        let data_file = build_unpartitioned_data_file();

        // Build the index on a runtime that is then DESTROYED, cancelling the populate task and
        // dropping its future. Done on a separate thread because dropping a runtime from inside
        // an async context panics.
        let (index, _tx) = std::thread::spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build the throwaway runtime that hosts the populate task");
            let (index, tx) = runtime.block_on(async {
                let (index, tx) = DeleteFileIndex::new();
                // Let the populate task reach its `collect()` await, past the point where its
                // drop guard is armed.
                for _ in 0..4 {
                    tokio::task::yield_now().await;
                }
                (index, tx)
            });
            drop(runtime);
            (index, tx)
        })
        .join()
        .expect("the runtime-teardown thread must not panic");

        let error = tokio::time::timeout(
            Duration::from_secs(5),
            index.get_deletes_for_data_file(&data_file, Some(0)),
        )
        .await
        .expect("a dead populate task must not hang the waiter")
        .expect_err("a dead populate task must surface a typed error");

        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert!(
            error.to_string().contains("populate task"),
            "the error must name the dead populate task, got: {error}"
        );
    }

    /// The terminal `Failed` state must be reported to EVERY subsequent caller, not just the one
    /// that was parked when the populate task died — the state can never advance again.
    #[tokio::test]
    async fn test_failed_index_state_is_terminal_for_later_callers() {
        let (index, _tx) = DeleteFileIndex::new();
        let data_file = build_unpartitioned_data_file();

        {
            let mut guard = index.state.write().expect("write the index state");
            *guard = DeleteFileIndexState::Failed("populate task was cancelled".to_string());
        }

        for _ in 0..2 {
            let error = tokio::time::timeout(
                Duration::from_secs(5),
                index.get_deletes_for_data_file(&data_file, Some(0)),
            )
            .await
            .expect("a failed index must answer immediately")
            .expect_err("a failed index must surface a typed error");
            assert_eq!(error.kind(), ErrorKind::Unexpected);
            assert!(
                error.to_string().contains("populate task was cancelled"),
                "the error must carry the recorded reason, got: {error}"
            );
        }
    }

    fn build_unpartitioned_data_file() -> DataFile {
        DataFileBuilder::default()
            .file_path(format!("{}-data.parquet", Uuid::new_v4()))
            .file_format(DataFileFormat::Parquet)
            .content(DataContentType::Data)
            .record_count(100)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .file_size_in_bytes(100)
            .build()
            .unwrap()
    }

    fn build_partitioned_data_file(partition_value: &Struct, spec_id: i32) -> DataFile {
        DataFileBuilder::default()
            .file_path(format!("{}-data.parquet", Uuid::new_v4()))
            .file_format(DataFileFormat::Parquet)
            .content(DataContentType::Data)
            .record_count(100)
            .partition(partition_value.clone())
            .partition_spec_id(spec_id)
            .file_size_in_bytes(100)
            .build()
            .unwrap()
    }

    fn build_added_manifest_entry(data_seq_number: i64, file: &DataFile) -> ManifestEntry {
        ManifestEntry::builder()
            .status(ManifestStatus::Added)
            .sequence_number(data_seq_number)
            .data_file(file.clone())
            .build()
    }
}
