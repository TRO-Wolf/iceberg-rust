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

use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;
use crate::runtime::spawn;
use crate::scan::{DeleteFileContext, FileScanTaskDeleteFile};
use crate::spec::{DataContentType, DataFile, DataFileFormat, PrimitiveLiteral, Struct};
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
    /// The populate task terminated WITHOUT publishing an index — it unwound, or its future was
    /// dropped by a runtime teardown (whether parked at `collect()` or never polled at all).
    ///
    /// Terminal, exactly like [`DeleteFileIndexState::Populated`]: the populate task is the sole
    /// writer and runs once, so if it dies the state can never advance on its own. Without this
    /// variant the state would stay `Populating` forever and every scan parked in
    /// [`DeleteFileIndex::get_deletes_for_data_file`] would wait on a notification that can no
    /// longer be sent. The `String` is the reason, rendered into each waiter's typed error. This
    /// mirrors `EqDelState::Failed` in [`crate::arrow::delete_filter`].
    Failed(String),
}

/// Publishes the populate task's TERMINAL state and wakes the waiters.
///
/// Constructed in the `spawn` PRELUDE — not inside the `async move` block — so the populate
/// future CAPTURES an already-armed guard rather than arming it on its first poll. That
/// distinction is the whole guarantee: a future constructed-on-poll that is dropped before it is
/// ever polled runs no local destructors, so a runtime torn down between `spawn` and the first
/// poll would leave the state stranded at `Populating`.
///
/// [`PopulateGuard::publish`] disarms it on the success path. If that call is never reached,
/// `Drop` publishes [`DeleteFileIndexState::Failed`] instead, so every waiter reaches a terminal
/// state and gets a typed error rather than hanging forever. `Drop` therefore covers all three
/// ways the task can die without publishing: never polled, unwound (tokio drops the task's future
/// as the panic propagates), and cancelled/parked-future teardown.
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
    ///
    /// Respects an EXISTING terminal state: if another writer already moved the index out of
    /// `Populating` — [`DeleteFileIndex::mark_failed`] on a delete-entry processing error — the
    /// earlier terminal wins and this publish only disarms + re-notifies (harmless). Behavior-
    /// identical for the pre-existing single-writer paths, which both publish from `Populating`.
    fn publish(&mut self, terminal: DeleteFileIndexState) {
        {
            // Recover a poisoned guard rather than cascading the panic: recovering and completing
            // the transition is always the right move (a stranded `Populating` state would hang
            // every waiting scan on the notifier below).
            let mut guard = self
                .state
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if matches!(*guard, DeleteFileIndexState::Populating(_)) {
                *guard = terminal;
            }
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

/// Partition delete maps are nested `spec_id → partition → deletes` so lookup is
/// `get(spec_id).get(partition)` without cloning the partition `Struct` (FK2.3 / scout #15).
///
/// Java `DeleteFileIndex` / `PartitionMap` keys by the composite `(spec_id, partition)` — never by
/// partition tuple alone. A flat `HashMap<Struct, _>` forced a post-filter linear scan on `spec_id`
/// and, on a wrong-key bug, resurrects deletes onto data files of a different evolved spec that
/// share the same partition values. A flat `HashMap<(i32, Struct), _>` is correct but clones the
/// partition on every lookup; the nested form is equivalent and keeps the pre-FK2.3 zero-clone
/// `get(partition)` hot path.
type PartitionDeleteMap = HashMap<i32, HashMap<Struct, Vec<Arc<DeleteFileContext>>>>;

#[derive(Debug)]
struct PopulatedDeleteFileIndex {
    /// Global equality deletes (unpartitioned). Sorted ascending by data sequence number so
    /// [`applicable_eq_deletes`] can `partition_point` the applicable tail (Java
    /// `EqualityDeletes.filter` / `findStartIndex` shape).
    global_equality_deletes: Vec<Arc<DeleteFileContext>>,
    /// Equality deletes keyed by `(spec_id, partition)` via nested maps (FK2.3). Each list is
    /// seq-sorted.
    eq_deletes_by_partition: PartitionDeleteMap,
    /// Partition-scoped position deletes keyed by `(spec_id, partition)` via nested maps (FK2.3).
    /// Each list is seq-sorted.
    pos_deletes_by_partition: PartitionDeleteMap,
    /// FILE-SCOPED position deletes keyed by the data file they reference, mirroring Java
    /// `DeleteFileIndex.posDeletesByPath` (`Builder.add(Map<String, PositionDeletes>,
    /// PartitionMap<PositionDeletes>, DeleteFile)`: when
    /// `ContentFileUtil.referencedDataFileLocation(file)` is non-null the file goes here INSTEAD of
    /// into [`Self::pos_deletes_by_partition`]).
    ///
    /// Consulted by path ALONE — no spec condition, no partition condition (Java `findPathDeletes`
    /// = `posDeletesByPath.get(dataFile.location())`). Spark's default write granularity is FILE, so
    /// this is the common shape in Java-written merge-on-read tables; before this map existed such a
    /// delete was indexed by its own `(spec_id, partition)` and a data file whose partition or spec
    /// differed never found it — the masked rows silently resurrected.
    ///
    /// Each list is seq-sorted so [`applicable_pos_deletes`] can `partition_point` the applicable
    /// tail (Java `PositionDeletes.filter` / `findStartIndex`).
    pos_deletes_by_path: HashMap<String, Vec<Arc<DeleteFileContext>>>,
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

/// Sort a delete-file list by data sequence number ascending (`None` first — Option order).
/// Stable: equal sequences keep insert order so multi-delete same-seq fixtures stay deterministic.
fn sort_deletes_by_sequence(deletes: &mut [Arc<DeleteFileContext>]) {
    deletes.sort_by_key(|d| d.manifest_entry.sequence_number());
}

/// Equality-delete applicability: `delete_seq > data_seq` (Java strict-greater). Lists must be
/// sorted by [`sort_deletes_by_sequence`]. When `seq_num` is `None` (unit fixtures), every delete
/// applies.
fn applicable_eq_deletes(
    deletes: &[Arc<DeleteFileContext>],
    seq_num: Option<i64>,
) -> &[Arc<DeleteFileContext>] {
    let Some(data_seq) = seq_num else {
        return deletes;
    };
    // partition_point: first index where the predicate is false.
    // Predicate "not yet applicable" ≡ delete_seq is None or delete_seq <= data_seq.
    let idx = deletes.partition_point(|d| match d.manifest_entry.sequence_number() {
        None => true,
        Some(s) => s <= data_seq,
    });
    &deletes[idx..]
}

/// Position-delete applicability: `delete_seq >= data_seq` (Java `findStartIndex`). Lists must be
/// sorted by [`sort_deletes_by_sequence`]. When `seq_num` is `None` (unit fixtures), every delete
/// applies.
fn applicable_pos_deletes(
    deletes: &[Arc<DeleteFileContext>],
    seq_num: Option<i64>,
) -> &[Arc<DeleteFileContext>] {
    let Some(data_seq) = seq_num else {
        return deletes;
    };
    // Predicate "not yet applicable" ≡ delete_seq is None or delete_seq < data_seq.
    let idx = deletes.partition_point(|d| match d.manifest_entry.sequence_number() {
        None => true,
        Some(s) => s < data_seq,
    });
    &deletes[idx..]
}

/// Whether a delete file is a deletion vector. Java `ContentFileUtil.isDV` (L142-144):
/// `deleteFile.format() == FileFormat.PUFFIN`.
pub(crate) fn is_deletion_vector(data_file: &DataFile) -> bool {
    data_file.file_format() == DataFileFormat::Puffin
}

/// The single data file a delete file references, or `None` when it references more than one.
///
/// The Rust mirror of Java `ContentFileUtil.referencedDataFile(DeleteFile)` +
/// `referencedDataFileLocation` (1.10.0 bytecode-decoded), which is THE routing predicate for
/// position deletes: `DeleteFileIndex.Builder.add` sends a delete with a referenced data file into
/// the PATH-keyed map and every other delete into the `(spec, partition)`-keyed map. The same
/// predicate governs which deletes the [`RemoveDanglingDeleteFiles`] maintenance action may collect
/// (`crate::maintenance::remove_dangling_delete_files`), so it lives in exactly one place — a reader
/// and a collector that disagree about file-scoping is how a still-applicable delete gets deleted.
///
/// Three legs, in Java's order:
///
/// 1. **Equality deletes are never file-scoped.** Java returns null before looking at anything else
///    (`content() == EQUALITY_DELETES → null`): an equality delete matches by VALUE across a whole
///    partition, so a `file_path` bound on it would be meaningless.
/// 2. **The explicit back-reference wins.** `referenced_data_file` (spec field 143), when set, is
///    returned as-is. Deletion vectors always carry it; a PARQUET position delete essentially never
///    does — through Iceberg 1.11.0 the only writer that sets the field is `deletes.BaseDVFileWriter`
///    (the V3 DV writer), so this leg is dead for every Spark-written V2 table and leg 3 carries them
///    all. Independently bytecode-verified at 1.10.0 and 1.11.0 by the RePark consumer, 2026-07-25.
/// 3. **Otherwise derive it from the `file_path`-column bounds.** Java reads the lower and upper
///    bound of the reserved `file_path` column ([`RESERVED_FIELD_ID_DELETE_FILE_PATH`]) and returns
///    the decoded value only when BOTH exist and are EQUAL — equal bounds mean every row in the
///    delete file names the same data file. This leg is not an optimization: Java's own
///    `PositionDeleteWriter.close()` never sets `referenced_data_file`, it only preserves those
///    bounds (`metrics()` strips them once a second referenced file appears), so leg 3 is how
///    virtually every Java-written file-granularity position delete is recognised. Implementing only
///    leg 2 leaves that entire class unrouted — and no fixture that sets the field can detect it.
///
/// Bounds that are absent, unequal, or not string-typed leave the delete partition-scoped, exactly
/// as in Java (which compares the raw bound `ByteBuffer`s and decodes with the `file_path` column's
/// string type). Bound TRUNCATION cannot forge a match: a truncated lower bound is shortened and the
/// matching upper bound is rounded UP, so the two are equal only when both are the full value.
pub(crate) fn referenced_data_file_location(delete_file: &DataFile) -> Option<String> {
    if delete_file.content_type() == DataContentType::EqualityDeletes {
        return None;
    }

    if let Some(referenced) = delete_file.referenced_data_file() {
        return Some(referenced);
    }

    let lower = delete_file
        .lower_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)?;
    let upper = delete_file
        .upper_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)?;

    match (lower.literal(), upper.literal()) {
        (PrimitiveLiteral::String(lower), PrimitiveLiteral::String(upper)) if lower == upper => {
            Some(lower.clone())
        }
        _ => None,
    }
}

/// Backpressure buffer capacity for delete-file contexts streamed into the index populate
/// task. Sized to absorb a burst of delete entries without stalling concurrent manifest
/// processing under typical scan concurrency; not a hard limit on total delete files (the
/// receiver drains the stream into a `Vec` before indexing).
const DELETE_FILE_INDEX_CHANNEL_CAPACITY: usize = 1024;

impl DeleteFileIndex {
    /// Move a still-`Populating` index to `Failed` and wake every parked waiter (review rider,
    /// 2026-08-03). Used by the plan path when DELETE-ENTRY processing errors: without this the
    /// entry error only reaches the task channel — the delete senders drop, the populate task
    /// sees a normal end-of-channel, and a PARTIAL delete set publishes as `Populated`. Data
    /// tasks streamed before the consumer observes the `Err` would read with missing deletes —
    /// silent row resurrection for an early-terminating (e.g. LIMIT-k) consumer. Java plans
    /// deletes before emitting any task; this restores fail-before-results under FK2.2's
    /// concurrent planning.
    ///
    /// No-op when the state is already terminal (`Populated` / `Failed`): the first terminal
    /// state wins, matching [`PopulateGuard::publish`]'s respect-terminal rule. Callers that
    /// need the failure to win DETERMINISTICALLY must call this while they still hold a live
    /// delete-channel sender — the populate task cannot publish before the channel closes, so
    /// `Failed` is guaranteed to land first.
    pub(crate) fn mark_failed(&self, reason: &str) {
        let notify = {
            let mut guard = self
                .state
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match &*guard {
                DeleteFileIndexState::Populating(notify) => {
                    let notify = notify.clone();
                    *guard = DeleteFileIndexState::Failed(format!(
                        "the delete file index was marked failed before population completed: \
                         {reason}"
                    ));
                    Some(notify)
                }
                _ => None,
            }
        };
        if let Some(notify) = notify {
            notify.notify_waiters();
        }
    }

    /// create a new `DeleteFileIndex` along with the sender that populates it with delete files
    pub(crate) fn new() -> (DeleteFileIndex, Sender<DeleteFileContext>) {
        let (tx, rx) = channel(DELETE_FILE_INDEX_CHANNEL_CAPACITY);
        let notify = Arc::new(Notify::new());
        let state = Arc::new(RwLock::new(DeleteFileIndexState::Populating(
            notify.clone(),
        )));
        let delete_file_stream = rx.boxed();

        spawn({
            // Armed HERE, in the prelude, so the future below CAPTURES the guard instead of
            // constructing it on its first poll. Dropping the future then always runs the guard's
            // `Drop` — including when the future is dropped before it is ever polled (a runtime
            // torn down between this `spawn` and the first poll), which a guard constructed inside
            // the `async move` block would miss entirely.
            let mut guard = PopulateGuard::new(state.clone(), notify);

            async move {
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
    /// 3. A FILE-SCOPED position delete (one with a derivable referenced data file — see
    ///    [`referenced_data_file_location`]) is added to `pos_deletes_by_path`, keyed by that file.
    /// 4. Otherwise, the delete file is added to one of two hash maps based on its content type.
    fn new(files: Vec<DeleteFileContext>) -> PopulatedDeleteFileIndex {
        let mut eq_deletes_by_partition: PartitionDeleteMap = HashMap::default();
        let mut pos_deletes_by_partition: PartitionDeleteMap = HashMap::default();
        let mut pos_deletes_by_path: HashMap<String, Vec<Arc<DeleteFileContext>>> =
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
                DataContentType::PositionDeletes => {
                    // Java `DeleteFileIndex.Builder.add(Map<String, PositionDeletes>,
                    // PartitionMap<PositionDeletes>, DeleteFile)`: a position delete with a
                    // derivable referenced data file is keyed by that PATH, otherwise by
                    // `(spec_id, partition)`. The two are EXCLUSIVE — indexing a file-scoped delete
                    // in both maps would attach it to every sibling data file in its partition and
                    // return it TWICE for the file it actually references.
                    if let Some(referenced_data_file) =
                        referenced_data_file_location(arc_ctx.manifest_entry.data_file())
                    {
                        pos_deletes_by_path
                            .entry(referenced_data_file)
                            .or_default()
                            .push(arc_ctx);
                        return;
                    }
                    &mut pos_deletes_by_partition
                }
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

            // FK2.3: nested `(spec_id → partition)` — DeleteFileContext's manifest spec id matches
            // the lookup key `data_file.partition_spec_id` (what the pre-FK2.3 linear post-filter
            // compared). Partition is cloned only on first insert into a bucket, not on lookup.
            destination_map
                .entry(arc_ctx.partition_spec_id)
                .or_default()
                .entry(partition.clone())
                .or_default()
                .push(arc_ctx);
        });

        // Sort once at build time so lookup can `partition_point` the applicable tail
        // (Java `EqualityDeletes` / `PositionDeletes` keep seq-sorted lists + `findStartIndex`).
        sort_deletes_by_sequence(&mut global_equality_deletes);
        for by_partition in eq_deletes_by_partition.values_mut() {
            for list in by_partition.values_mut() {
                sort_deletes_by_sequence(list);
            }
        }
        for by_partition in pos_deletes_by_partition.values_mut() {
            for list in by_partition.values_mut() {
                sort_deletes_by_sequence(list);
            }
        }
        for list in pos_deletes_by_path.values_mut() {
            sort_deletes_by_sequence(list);
        }

        PopulatedDeleteFileIndex {
            global_equality_deletes,
            eq_deletes_by_partition,
            pos_deletes_by_partition,
            pos_deletes_by_path,
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

        // Global equality: seq-sorted + partition_point for `delete_seq > data_seq`.
        for delete in applicable_eq_deletes(&self.global_equality_deletes, seq_num) {
            results.push(delete.as_ref().into());
        }

        // FK2.3: nested `(spec_id → partition)` lookup — no post-filter linear scan on
        // spec_id, no partition Struct clone on the hot path.
        if let Some(by_partition) = self
            .eq_deletes_by_partition
            .get(&data_file.partition_spec_id)
            && let Some(deletes) = by_partition.get(data_file.partition())
        {
            for delete in applicable_eq_deletes(deletes, seq_num) {
                results.push(delete.as_ref().into());
            }
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

        // Java `findPosPartitionDeletes` (L304-328): the `(spec_id, partition)`-keyed position
        // deletes — the ones with no derivable referenced data file. Seq-sorted +
        // `partition_point` for `delete_seq >= data_seq`.
        if let Some(by_partition) = self
            .pos_deletes_by_partition
            .get(&data_file.partition_spec_id)
            && let Some(deletes) = by_partition.get(data_file.partition())
        {
            for delete in applicable_pos_deletes(deletes, seq_num) {
                results.push(delete.as_ref().into());
            }
        }

        // Java `findPathDeletes` (L355-377): the FILE-SCOPED position deletes, looked up by the data
        // file's LOCATION and filtered ONLY by sequence number — `posDeletesByPath.get(dataFile
        // .location())` then `PositionDeletes.filter(seq)`. There is deliberately NO spec condition
        // and NO partition condition here: the delete names this exact file, so the partition tuple
        // it happens to be stamped with is irrelevant (and is routinely a different spec's — Spark's
        // default write granularity is FILE). The sequence rule is the same `>=` as the partition
        // map (Java `findStartIndex` keeps `delete_seq >= data_seq`); only the KEY changes.
        //
        // Appended after the partition-keyed deletes to mirror Java's
        // `concat(global, eqPartition, posPartition, posPath)` result order.
        if let Some(deletes) = self.pos_deletes_by_path.get(data_file.file_path()) {
            for delete in applicable_pos_deletes(deletes, seq_num) {
                results.push(delete.as_ref().into());
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use uuid::Uuid;

    use super::*;
    use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Datum, Literal, ManifestEntry,
        ManifestStatus, Struct,
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

    /// A PARTITION-scoped parquet position delete: no `referenced_data_file`, no `file_path`-column
    /// bounds — nothing from which Java's `ContentFileUtil.referencedDataFile` could derive a single
    /// referenced data file, so it is indexed by `(spec_id, partition)`.
    fn build_partitioned_pos_delete(partition: &Struct, spec_id: i32) -> DataFile {
        DataFileBuilder::default()
            .file_path(format!("{}-pos-delete.parquet", Uuid::new_v4()))
            .file_format(DataFileFormat::Parquet)
            .content(DataContentType::PositionDeletes)
            .record_count(1)
            .partition(partition.clone())
            .partition_spec_id(spec_id)
            .file_size_in_bytes(100)
            .build()
            .unwrap()
    }

    /// A FILE-scoped parquet position delete carrying the explicit `referenced_data_file`
    /// back-reference (leg (a) of Java `ContentFileUtil.referencedDataFile`: the field, when set,
    /// wins outright).
    fn build_file_scoped_pos_delete(
        referenced: &str,
        partition: &Struct,
        spec_id: i32,
    ) -> DataFile {
        DataFileBuilder::default()
            .file_path(format!("{}-pos-delete.parquet", Uuid::new_v4()))
            .file_format(DataFileFormat::Parquet)
            .content(DataContentType::PositionDeletes)
            .record_count(1)
            .referenced_data_file(Some(referenced.to_string()))
            .partition(partition.clone())
            .partition_spec_id(spec_id)
            .file_size_in_bytes(100)
            .build()
            .unwrap()
    }

    /// A parquet position delete carrying ONLY the `file_path`-column lower/upper BOUNDS — the shape
    /// Java's `PositionDeleteWriter` actually emits (its `close()` never calls
    /// `withReferencedDataFile`; `metrics()` keeps the `file_path` bounds whenever the writer saw at
    /// most one referenced data file — 1.10.0 bytecode). Leg (b) of
    /// `ContentFileUtil.referencedDataFile`.
    fn build_pos_delete_with_path_bounds(
        lower: &str,
        upper: &str,
        partition: &Struct,
        spec_id: i32,
    ) -> DataFile {
        DataFileBuilder::default()
            .file_path(format!("{}-pos-delete.parquet", Uuid::new_v4()))
            .file_format(DataFileFormat::Parquet)
            .content(DataContentType::PositionDeletes)
            .record_count(1)
            .lower_bounds(HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string(lower),
            )]))
            .upper_bounds(HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string(upper),
            )]))
            .partition(partition.clone())
            .partition_spec_id(spec_id)
            .file_size_in_bytes(100)
            .build()
            .unwrap()
    }

    /// Index `deletes` (each paired with its data sequence number) under `spec_id` and return the
    /// paths the index hands back for `data_file` at `seq_num`.
    fn applied_paths(
        deletes: Vec<(i64, DataFile)>,
        spec_id: i32,
        data_file: &DataFile,
        seq_num: i64,
    ) -> Vec<String> {
        let contexts: Vec<DeleteFileContext> = deletes
            .iter()
            .map(|(seq, file)| DeleteFileContext {
                manifest_entry: build_added_manifest_entry(*seq, file).into(),
                partition_spec_id: spec_id,
            })
            .collect();
        PopulatedDeleteFileIndex::new(contexts)
            .get_deletes_for_data_file(data_file, Some(seq_num))
            .expect("the index lookup must succeed")
            .into_iter()
            .map(|file| file.file_path)
            .collect()
    }

    // =========================================================================================
    // PATH-KEYED position-delete routing (Java `DeleteFileIndex.findPathDeletes` +
    // `Builder.add(Map<String, PositionDeletes>, PartitionMap<...>, DeleteFile)`)
    // =========================================================================================

    /// Risk pinned (WG4b): Java routes a FILE-SCOPED position delete — one whose referenced data
    /// file is derivable — into `posDeletesByPath`, which `findPathDeletes` consults with the data
    /// file's LOCATION ALONE: no spec condition, no partition condition (1.10.0 bytecode,
    /// `findPathDeletes` = `posDeletesByPath.get(dataFile.location())`). Leg (a): the explicit
    /// `referenced_data_file` field.
    ///
    /// The delete here is stamped spec 0 with an EMPTY partition while the data file lives in spec 1
    /// partition `{100}` — the shape a writer that defaults the spec stamp produces. Before the path
    /// map existed the fork indexed it by `(0, {})`, no lookup ever reached it, and the rows it masks
    /// silently resurrected.
    ///
    /// MUTATION: routing file-scoped deletes back into the partition map (or dropping the path
    /// lookup) empties this result (RED).
    #[test]
    fn test_file_scoped_pos_delete_applies_regardless_of_partition_and_spec() {
        let data_file =
            build_partitioned_data_file(&Struct::from_iter([Some(Literal::long(100))]), 1);
        let delete = build_file_scoped_pos_delete(data_file.file_path(), &Struct::empty(), 0);
        let delete_path = delete.file_path().to_string();

        let applied = applied_paths(vec![(2, delete)], 0, &data_file, 1);

        assert_eq!(
            applied,
            vec![delete_path],
            "a file-scoped position delete must apply to the data file it references even though \
             its spec id (0 vs 1) and partition tuple ({{}} vs {{100}}) both differ"
        );
    }

    /// Risk pinned (WG4b leg (b)): Java's `PositionDeleteWriter` never sets `referenced_data_file`
    /// — it keeps the `file_path`-column BOUNDS instead, and `ContentFileUtil.referencedDataFile`
    /// derives the path from them when lower == upper (1.10.0 bytecode: read
    /// `lowerBounds().get(PATH_ID)` / `upperBounds().get(PATH_ID)`, return null unless both exist and
    /// `lower.equals(upper)`, then `Conversions.fromByteBuffer(StringType, lower)`). Implementing
    /// only the field leg leaves EVERY Java-written file-granularity position delete unrouted — a
    /// fixture that sets the field cannot detect it.
    ///
    /// MUTATION: dropping the bounds leg (field-only routing) sends this delete to the partition map
    /// under `(0, {})`, which the `(1, {100})` data file never consults (RED).
    #[test]
    fn test_pos_delete_with_equal_path_bounds_is_file_scoped() {
        let data_file =
            build_partitioned_data_file(&Struct::from_iter([Some(Literal::long(100))]), 1);
        let delete = build_pos_delete_with_path_bounds(
            data_file.file_path(),
            data_file.file_path(),
            &Struct::empty(),
            0,
        );
        let delete_path = delete.file_path().to_string();

        let applied = applied_paths(vec![(2, delete)], 0, &data_file, 1);

        assert_eq!(
            applied,
            vec![delete_path],
            "equal file_path lower/upper bounds identify ONE referenced data file, so the delete is \
             file-scoped and applies across the spec/partition mismatch"
        );
    }

    /// Risk pinned: UNEQUAL `file_path` bounds mean the delete spans MORE than one data file, so
    /// Java derives no referenced file and keeps it PARTITION-scoped (`lower.equals(upper)` is the
    /// only gate). It must therefore still obey the partition + spec condition: it reaches a data
    /// file in its own partition and NOT one in another, even though that other file's path equals
    /// the lower bound.
    ///
    /// MUTATION: keying the path map off the lower bound alone (dropping the `lower == upper`
    /// condition) makes the mismatched-partition file receive it (RED on the second assertion).
    #[test]
    fn test_pos_delete_with_unequal_path_bounds_stays_partition_scoped() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let same_partition_file = build_partitioned_data_file(&partition, 1);
        let other_partition_file =
            build_partitioned_data_file(&Struct::from_iter([Some(Literal::long(200))]), 1);

        let delete = build_pos_delete_with_path_bounds(
            other_partition_file.file_path(),
            same_partition_file.file_path(),
            &partition,
            1,
        );
        let delete_path = delete.file_path().to_string();

        assert_eq!(
            applied_paths(vec![(2, delete.clone())], 1, &same_partition_file, 1),
            vec![delete_path],
            "a multi-file position delete stays partition-scoped and applies within its partition"
        );
        assert!(
            applied_paths(vec![(2, delete)], 1, &other_partition_file, 1).is_empty(),
            "it must NOT be treated as file-scoped for the file its LOWER bound names — Java \
             derives a referenced file only when lower == upper"
        );
    }

    /// Risk pinned: Java's routing is EXCLUSIVE — `Builder.add` puts a file-scoped delete in the
    /// path map INSTEAD of the partition map (`if (path != null) … else …`, 1.10.0 bytecode). A
    /// sibling data file in the same partition must therefore not receive it, and must not receive
    /// it twice.
    ///
    /// MUTATION: indexing file-scoped deletes into BOTH maps makes the sibling receive it (RED on
    /// the sibling assertion) and duplicates it for the referenced file (RED on the first).
    #[test]
    fn test_file_scoped_pos_delete_is_not_applied_to_a_sibling_in_the_same_partition() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let referenced_file = build_partitioned_data_file(&partition, 1);
        let sibling_file = build_partitioned_data_file(&partition, 1);
        let delete = build_file_scoped_pos_delete(referenced_file.file_path(), &partition, 1);
        let delete_path = delete.file_path().to_string();

        assert_eq!(
            applied_paths(vec![(2, delete.clone())], 1, &referenced_file, 1),
            vec![delete_path],
            "the referenced data file receives the file-scoped delete exactly once"
        );
        assert!(
            applied_paths(vec![(2, delete)], 1, &sibling_file, 1).is_empty(),
            "a sibling in the SAME partition must not receive a file-scoped delete — Java indexes \
             it by path INSTEAD of by partition"
        );
    }

    /// CONTROL (must hold before and after the path map exists): a genuinely partition-scoped
    /// position delete — no referenced field, no path bounds — still requires BOTH the partition
    /// tuple and the spec id to match. This is the pin that stops "route everything by path" from
    /// passing: it has no path to be routed by.
    #[test]
    fn test_partition_scoped_pos_delete_still_requires_matching_partition_and_spec() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let delete = build_partitioned_pos_delete(&partition, 1);
        let delete_path = delete.file_path().to_string();

        assert_eq!(
            applied_paths(
                vec![(2, delete.clone())],
                1,
                &build_partitioned_data_file(&partition, 1),
                1
            ),
            vec![delete_path],
            "same partition + same spec: it applies"
        );
        assert!(
            applied_paths(
                vec![(2, delete.clone())],
                1,
                &build_partitioned_data_file(&Struct::from_iter([Some(Literal::long(200))]), 1),
                1
            )
            .is_empty(),
            "different partition tuple: it must NOT apply"
        );
        assert!(
            applied_paths(
                vec![(2, delete)],
                1,
                &build_partitioned_data_file(&partition, 2),
                1
            )
            .is_empty(),
            "same partition tuple, different spec id: it must NOT apply"
        );
    }

    /// Risk pinned: Java's `ContentFileUtil.referencedDataFile` returns null for EQUALITY_DELETES
    /// BEFORE it looks at either the field or the bounds (the first branch, 1.10.0 bytecode). An
    /// equality delete is never file-scoped, whatever its metrics say — it stays a partition (or
    /// global) delete, and applies STRICTLY to lower-sequence data.
    ///
    /// MUTATION (content-BLIND file-scoping — hoisting the path routing above the content-type match
    /// AND dropping the helper's equality early-return): this delete lands in the POSITION path map,
    /// where its partition peers never find it (RED on the second assertion). The early-return alone
    /// is not observable HERE, because the index consults the helper only inside the
    /// `PositionDeletes` arm; it is load-bearing on the maintenance side, pinned by
    /// `remove_dangling_delete_files`'s
    /// `test_equality_delete_with_path_bounds_is_judged_by_min_seq_not_by_reference`.
    #[test]
    fn test_equality_delete_with_path_bounds_is_never_file_scoped() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let data_file = build_partitioned_data_file(&partition, 1);

        let mut eq_delete = build_partitioned_eq_delete(&partition, 1);
        eq_delete.lower_bounds = HashMap::from([(
            RESERVED_FIELD_ID_DELETE_FILE_PATH,
            Datum::string(data_file.file_path()),
        )]);
        eq_delete.upper_bounds = HashMap::from([(
            RESERVED_FIELD_ID_DELETE_FILE_PATH,
            Datum::string(data_file.file_path()),
        )]);
        let eq_delete_path = eq_delete.file_path().to_string();

        assert_eq!(
            applied_paths(vec![(4, eq_delete.clone())], 1, &data_file, 1),
            vec![eq_delete_path.clone()],
            "the equality delete applies to its partition's data file"
        );
        assert_eq!(
            applied_paths(
                vec![(4, eq_delete)],
                1,
                &build_partitioned_data_file(&partition, 1),
                1
            ),
            vec![eq_delete_path],
            "and to every OTHER data file in the same partition — it was not diverted into the \
             file-scoped position map"
        );
    }

    /// Risk pinned: the path map is sequence-filtered exactly like the partition map — Java's
    /// `findPathDeletes` calls `PositionDeletes.filter(seq)`, whose `findStartIndex` keeps the files
    /// with `delete_seq >= data_seq` (1.10.0 bytecode). Only the partition/spec condition is dropped
    /// by the path key, never the sequence rule.
    ///
    /// MUTATION: dropping the sequence filter on the path lookup makes the seq-3 delete apply to the
    /// seq-4 data file (RED).
    #[test]
    fn test_file_scoped_pos_delete_is_sequence_filtered_like_the_partition_map() {
        let data_file =
            build_partitioned_data_file(&Struct::from_iter([Some(Literal::long(100))]), 1);
        let delete = build_file_scoped_pos_delete(data_file.file_path(), &Struct::empty(), 0);
        let delete_path = delete.file_path().to_string();

        assert_eq!(
            applied_paths(vec![(4, delete.clone())], 0, &data_file, 4),
            vec![delete_path.clone()],
            "delete_seq == data_seq (4 == 4): a position delete applies at the boundary"
        );
        assert_eq!(
            applied_paths(vec![(4, delete.clone())], 0, &data_file, 3),
            vec![delete_path],
            "delete_seq > data_seq (4 > 3): it applies"
        );
        assert!(
            applied_paths(vec![(3, delete)], 0, &data_file, 4).is_empty(),
            "delete_seq < data_seq (3 < 4): it must NOT apply"
        );
    }

    /// Risk pinned: when a data file has a DELETION VECTOR, Java's `forDataFile` returns
    /// {global eq, partition eq, DV} and never calls `findPosPartitionDeletes` OR `findPathDeletes`
    /// (1.10.0 bytecode: both are inside the `dv == null` branch). The DV is the complete
    /// position-delete state for that file; also returning a superseded parquet delete would
    /// re-apply deletes the DV already accounts for.
    ///
    /// MUTATION: consulting the path map before/alongside the DV branch returns the parquet delete
    /// too (RED).
    #[test]
    fn test_dv_supersedes_a_file_scoped_position_delete() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let data_file = build_partitioned_data_file(&partition, 1);
        let dv = build_partitioned_deletion_vector(data_file.file_path(), &partition, 1);
        let dv_path = dv.file_path().to_string();

        let applied = applied_paths(
            vec![
                (
                    2,
                    build_file_scoped_pos_delete(data_file.file_path(), &partition, 1),
                ),
                (2, dv),
            ],
            1,
            &data_file,
            1,
        );

        assert_eq!(
            applied,
            vec![dv_path],
            "the DV supersedes the file-scoped parquet position delete, exactly as it supersedes a \
             partition-scoped one"
        );
    }

    /// Risk pinned: the result ORDER mirrors Java's `concat(global, eqPartition, posPartition,
    /// posPath)` (1.10.0 bytecode). Order is observable — `FileScanTask.deletes` is serialized and
    /// compared against Java in the interop suites — so it is pinned rather than left incidental.
    #[test]
    fn test_result_order_matches_java_concat_global_eq_partition_pos_path() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let data_file = build_partitioned_data_file(&partition, 1);

        let global_eq = build_partitioned_eq_delete(&Struct::empty(), 1);
        let partition_eq = build_partitioned_eq_delete(&partition, 1);
        let partition_pos = build_partitioned_pos_delete(&partition, 1);
        let path_pos = build_file_scoped_pos_delete(data_file.file_path(), &Struct::empty(), 1);
        let expected = vec![
            global_eq.file_path().to_string(),
            partition_eq.file_path().to_string(),
            partition_pos.file_path().to_string(),
            path_pos.file_path().to_string(),
        ];

        let applied = applied_paths(
            vec![
                (2, path_pos),
                (2, partition_pos),
                (2, partition_eq),
                (2, global_eq),
            ],
            1,
            &data_file,
            1,
        );

        assert_eq!(
            applied, expected,
            "global equality, then partition equality, then partition position, then path position"
        );
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

        // `DataFileBuilder` now refuses this shape, so build a valid DV and strip the field. Only the
        // manifest READ path can produce it — it decodes into a struct literal, not the builder.
        let mut invalid_dv = DataFileBuilder::default()
            .file_path("orphan-deletes.puffin".to_string())
            .file_format(DataFileFormat::Puffin)
            .content(DataContentType::PositionDeletes)
            .record_count(2)
            .partition(partition.clone())
            .partition_spec_id(spec_id)
            .file_size_in_bytes(100)
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40))
            .referenced_data_file(Some("placeholder.parquet".to_string()))
            .build()
            .unwrap();
        invalid_dv.referenced_data_file = None;

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

    /// Risk pinned (audit SAF-007, Critic probe P1b): the populate future can be dropped BEFORE IT
    /// IS EVER POLLED — a runtime torn down between `spawn` and the first poll. A future dropped
    /// unpolled runs no local destructors, so a guard constructed *inside* the `async move` block
    /// would never exist and the state would strand at `Populating`. The guard is therefore
    /// constructed in the `spawn` prelude and CAPTURED by the future.
    ///
    /// MUTATION: moving `PopulateGuard::new(...)` back inside the `async move` block leaves this
    /// waiter with no terminal state and the timeout below fires (RED), while the parked-future
    /// test above still passes — which is exactly why this pin is separate from it.
    #[tokio::test]
    async fn test_never_polled_populate_task_yields_a_typed_error_not_a_hang() {
        let data_file = build_unpartitioned_data_file();

        let (index, _tx) = std::thread::spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build the throwaway runtime that hosts the populate task");
            // ZERO yields: `block_on` returns as soon as `new()` does, so the populate future is
            // queued but never polled before the runtime below is destroyed.
            let (index, tx) = runtime.block_on(async { DeleteFileIndex::new() });
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
        .expect("a never-polled populate task must not hang the waiter")
        .expect_err("a never-polled populate task must surface a typed error");

        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert!(
            error.to_string().contains("populate task"),
            "the error must name the dead populate task, got: {error}"
        );
    }

    /// Risk pinned (audit SAF-007, Critic probe P2): a populate task that UNWINDS must leave the
    /// index in the terminal `Failed` state, not stranded at `Populating`.
    ///
    /// The panic is raised in a task holding a [`PopulateGuard`] over the real index's state
    /// rather than inside the real populate task: that task's body has no reachable panic source
    /// today (`PopulatedDeleteFileIndex::new` handles every `DataContentType` without indexing or
    /// unwrapping), so injecting one would need a test-only seam in production code. What is
    /// pinned is the property that matters — an unwind past a live guard publishes `Failed` and
    /// wakes the waiters.
    #[tokio::test]
    async fn test_unwinding_populate_task_yields_a_typed_error_not_a_hang() {
        let (index, _tx) = DeleteFileIndex::new();
        let data_file = build_unpartitioned_data_file();

        let notifier = {
            let guard = index.state.read().expect("read the index state");
            match &*guard {
                DeleteFileIndexState::Populating(notifier) => notifier.clone(),
                other => panic!("expected a populating index, got {other:?}"),
            }
        };

        let state = index.state.clone();
        let join_error = spawn(async move {
            let _guard = PopulateGuard::new(state, notifier);
            panic!("simulated populate-task unwind");
        })
        .try_join()
        .await
        .expect_err("the simulated populate task must have unwound");
        assert_eq!(join_error.kind(), ErrorKind::Unexpected);

        let error = tokio::time::timeout(
            Duration::from_secs(5),
            index.get_deletes_for_data_file(&data_file, Some(0)),
        )
        .await
        .expect("an unwound populate task must not hang the waiter")
        .expect_err("an unwound populate task must surface a typed error");

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

    // =========================================================================================
    // FK2.2 — overlap hang classes (inject-only, bounded timeouts; no loom / stress harness)
    // =========================================================================================

    /// FK2.2 hang class: lost-wakeup under the concurrent plan path.
    ///
    /// Data-entry processing parks on `get_deletes_for_data_file` while delete-entry processing
    /// populates. The full public await path (not just `lookup_or_arm`) must wake when publish
    /// races after the waiter has armed. Inject-only: keep the real populate task parked (sender
    /// alive), arm via a concurrent waiter, publish through `PopulateGuard`, assert the waiter
    /// completes under a generous timeout.
    ///
    /// MUTATION: arming the `Notified` AFTER releasing the read lock (raw `Arc<Notify>` +
    /// `.notified()` at the await site) loses the wakeup when publish races into that window —
    /// the timeout below fires (RED).
    #[tokio::test]
    async fn test_fk2_2_get_deletes_does_not_lose_wakeup_when_publish_races() {
        let (index, _tx) = DeleteFileIndex::new();
        let data_file = build_unpartitioned_data_file();

        // Deterministic arm FIRST (production handshake under the read lock) — yield-races are
        // not a substitute for proving the Notified exists before publish.
        let notified = match index
            .lookup_or_arm(&data_file, Some(0))
            .expect("arming a populating index must not error")
        {
            IndexLookup::Wait(n) => n,
            IndexLookup::Ready(_) => {
                panic!("the index must still be populating while the sender is alive")
            }
        };

        // Concurrent full-path waiter (second arm via `get_deletes_for_data_file`).
        let waiter_index = index.clone();
        let waiter = tokio::spawn(async move {
            let probe = build_unpartitioned_data_file();
            waiter_index
                .get_deletes_for_data_file(&probe, Some(0))
                .await
        });
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }

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
            .expect("publish after arm must wake the Notified, not be lost");

        let deletes = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("publish after arm must wake get_deletes_for_data_file, not hang")
            .expect("waiter task must not panic")
            .expect("the published empty index must succeed");
        assert!(
            deletes.is_empty(),
            "empty published index must yield no deletes"
        );
    }

    /// FK2.2 hang class: failed-populate under the concurrent plan path.
    ///
    /// If the populate task dies without publishing (sender dropped after a Forced `Failed`
    /// injection, or real teardown), every concurrent data-entry waiter must get a typed error
    /// under a bounded timeout — never hang on a `Notify` that can no longer fire.
    ///
    /// MUTATION: removing the `Failed` terminal state leaves waiters parked forever (timeout RED).
    #[tokio::test]
    async fn test_fk2_2_failed_populate_wakes_concurrent_waiters_with_typed_error() {
        let (index, _tx) = DeleteFileIndex::new();

        let waiter_a = {
            let index = index.clone();
            tokio::spawn(async move {
                let probe = build_unpartitioned_data_file();
                index.get_deletes_for_data_file(&probe, Some(0)).await
            })
        };
        let waiter_b = {
            let index = index.clone();
            tokio::spawn(async move {
                let probe = build_unpartitioned_data_file();
                index.get_deletes_for_data_file(&probe, Some(0)).await
            })
        };

        for _ in 0..16 {
            tokio::task::yield_now().await;
        }

        // Inject Failed via the production publisher (same write-lock-then-notify handshake as a
        // dead populate task's PopulateGuard::Drop).
        let notifier = {
            let guard = index.state.read().expect("read the index state");
            match &*guard {
                DeleteFileIndexState::Populating(notifier) => notifier.clone(),
                other => panic!("expected a populating index, got {other:?}"),
            }
        };
        PopulateGuard::new(index.state.clone(), notifier).publish(DeleteFileIndexState::Failed(
            "the delete file index populate task terminated before publishing an index \
             (it panicked, or the runtime was shut down)"
                .to_string(),
        ));

        for (label, waiter) in [("a", waiter_a), ("b", waiter_b)] {
            let error = tokio::time::timeout(Duration::from_secs(5), waiter)
                .await
                .unwrap_or_else(|_| panic!("waiter {label} must not hang on Failed populate"))
                .unwrap_or_else(|_| panic!("waiter {label} task must not panic"))
                .expect_err("Failed populate must surface a typed error");
            assert_eq!(error.kind(), ErrorKind::Unexpected);
            assert!(
                error.to_string().contains("populate task"),
                "waiter {label} error must name the dead populate task, got: {error}"
            );
        }
    }

    /// Review rider: `mark_failed` while `Populating` wakes a parked waiter with a typed error
    /// carrying the injected cause (a delete-entry processing error must fail the index, not
    /// let a partial delete set publish as `Populated`).
    #[tokio::test]
    async fn test_mark_failed_wakes_waiter_with_typed_error() {
        let (index, tx) = DeleteFileIndex::new();

        let waiter = {
            let index = index.clone();
            tokio::spawn(async move {
                let probe = build_unpartitioned_data_file();
                index.get_deletes_for_data_file(&probe, Some(0)).await
            })
        };
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }

        // Mark while the delete channel is still open (the caller-holds-a-sender contract).
        index.mark_failed("injected delete-entry processing error");
        drop(tx);

        let error = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("waiter must not hang after mark_failed")
            .expect("waiter task must not panic")
            .expect_err("mark_failed must surface a typed error to parked waiters");
        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert!(
            error
                .to_string()
                .contains("injected delete-entry processing error"),
            "the error must carry the injected cause, got: {error}"
        );
    }

    /// Review rider — the RACE-ORDER property: `mark_failed` called while the caller still holds
    /// a live sender lands strictly BEFORE the populate task can publish; the populate task's
    /// later `Populated` publish must NOT clobber it (respect-terminal). Mutation: revert
    /// `PopulateGuard::publish` to the unconditional overwrite → this test REDs (the partial
    /// index would win and `get_deletes_for_data_file` would succeed).
    #[tokio::test]
    async fn test_mark_failed_wins_over_later_populate_publish() {
        let (index, tx) = DeleteFileIndex::new();

        // Failure first, while the channel is provably open.
        index.mark_failed("boom before channel close");
        // NOW let the populate task complete its collect and attempt Populated.
        drop(tx);
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }

        let probe = build_unpartitioned_data_file();
        let error = tokio::time::timeout(
            Duration::from_secs(5),
            index.get_deletes_for_data_file(&probe, Some(0)),
        )
        .await
        .expect("must not hang")
        .expect_err("Failed must win over the later Populated publish");
        assert!(
            error.to_string().contains("boom before channel close"),
            "the FIRST terminal state (Failed) must stick, got: {error}"
        );
    }

    /// Review rider — the no-op direction: `mark_failed` after a successful publish must not
    /// disturb a `Populated` index (first terminal state wins in both directions).
    #[tokio::test]
    async fn test_mark_failed_after_populated_is_noop() {
        let (index, tx) = DeleteFileIndex::new();
        drop(tx); // empty index populates immediately
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
        let probe = build_unpartitioned_data_file();
        tokio::time::timeout(
            Duration::from_secs(5),
            index.get_deletes_for_data_file(&probe, Some(0)),
        )
        .await
        .expect("must not hang")
        .expect("empty index must be Populated and queryable");

        index.mark_failed("too late — already populated");

        let deletes = tokio::time::timeout(
            Duration::from_secs(5),
            index.get_deletes_for_data_file(&probe, Some(0)),
        )
        .await
        .expect("must not hang")
        .expect("mark_failed after Populated must be a no-op");
        assert!(deletes.is_empty());
    }

    /// FK2.2 natural path: dropping the last delete-context sender closes the channel, the
    /// populate task publishes, and a concurrent waiter wakes with the collected deletes.
    #[tokio::test]
    async fn test_fk2_2_sender_drop_publishes_and_wakes_waiter() {
        let (index, mut tx) = DeleteFileIndex::new();
        let partition = Struct::from_iter([Some(Literal::long(100))]);
        let spec_id = 1;
        let data_file = build_partitioned_data_file(&partition, spec_id);
        let delete = build_partitioned_eq_delete(&partition, spec_id);
        let delete_path = delete.file_path().to_string();

        tx.try_send(DeleteFileContext {
            manifest_entry: build_added_manifest_entry(4, &delete).into(),
            partition_spec_id: spec_id,
        })
        .expect("channel has capacity for one delete context");

        let waiter_index = index.clone();
        let probe_partition = partition.clone();
        let waiter = tokio::spawn(async move {
            let probe = build_partitioned_data_file(&probe_partition, spec_id);
            waiter_index
                .get_deletes_for_data_file(&probe, Some(0))
                .await
        });

        for _ in 0..8 {
            tokio::task::yield_now().await;
        }

        // Dropping the last sender closes the channel → populate collects → publishes.
        drop(tx);

        let deletes = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("sender drop must publish and wake the waiter, not hang")
            .expect("waiter task must not panic")
            .expect("populated index lookup must succeed");
        let paths: Vec<String> = deletes.into_iter().map(|d| d.file_path).collect();
        assert_eq!(
            paths,
            vec![delete_path],
            "the delete sent before the sender drop must apply to the data file"
        );
        let _ = data_file;
    }

    // =========================================================================================
    // FK2.3 — multi-spec composite keys: wrong-key = delete resurrection
    // =========================================================================================

    /// FK2.3 bar: identical result sets across a multi-spec fixture that shares the same
    /// partition tuple under two specs. Each data file must receive ONLY its own spec's deletes;
    /// attaching the other spec's deletes is delete resurrection.
    ///
    /// MUTATION: keying partition maps by `Struct` alone (dropping `spec_id` from the key) and
    /// forgetting the post-filter makes BOTH data files receive ALL four deletes (RED on both
    /// exact-set asserts). Keying by `(spec_id, partition)` keeps the sets disjoint.
    #[test]
    fn test_fk2_3_multi_spec_identical_result_sets_no_cross_spec_resurrection() {
        let partition = Struct::from_iter([Some(Literal::long(100))]);

        let eq_s1 = build_partitioned_eq_delete(&partition, 1);
        let pos_s1 = build_partitioned_pos_delete(&partition, 1);
        let eq_s2 = build_partitioned_eq_delete(&partition, 2);
        let pos_s2 = build_partitioned_pos_delete(&partition, 2);

        let eq_s1_path = eq_s1.file_path().to_string();
        let pos_s1_path = pos_s1.file_path().to_string();
        let eq_s2_path = eq_s2.file_path().to_string();
        let pos_s2_path = pos_s2.file_path().to_string();

        let contexts = vec![
            DeleteFileContext {
                manifest_entry: build_added_manifest_entry(2, &eq_s1).into(),
                partition_spec_id: 1,
            },
            DeleteFileContext {
                manifest_entry: build_added_manifest_entry(2, &pos_s1).into(),
                partition_spec_id: 1,
            },
            DeleteFileContext {
                manifest_entry: build_added_manifest_entry(2, &eq_s2).into(),
                partition_spec_id: 2,
            },
            DeleteFileContext {
                manifest_entry: build_added_manifest_entry(2, &pos_s2).into(),
                partition_spec_id: 2,
            },
        ];
        let index = PopulatedDeleteFileIndex::new(contexts);

        let data_s1 = build_partitioned_data_file(&partition, 1);
        let data_s2 = build_partitioned_data_file(&partition, 2);

        let paths_s1: Vec<String> = index
            .get_deletes_for_data_file(&data_s1, Some(0))
            .expect("spec-1 lookup")
            .into_iter()
            .map(|d| d.file_path)
            .collect();
        let paths_s2: Vec<String> = index
            .get_deletes_for_data_file(&data_s2, Some(0))
            .expect("spec-2 lookup")
            .into_iter()
            .map(|d| d.file_path)
            .collect();

        assert_eq!(
            paths_s1,
            vec![eq_s1_path.clone(), pos_s1_path.clone()],
            "spec-1 data file must receive ONLY spec-1 deletes (eq then pos), never spec-2"
        );
        assert_eq!(
            paths_s2,
            vec![eq_s2_path.clone(), pos_s2_path.clone()],
            "spec-2 data file must receive ONLY spec-2 deletes (eq then pos), never spec-1"
        );
        assert!(
            !paths_s1.contains(&eq_s2_path) && !paths_s1.contains(&pos_s2_path),
            "cross-spec resurrection: spec-1 must not see any spec-2 delete"
        );
        assert!(
            !paths_s2.contains(&eq_s1_path) && !paths_s2.contains(&pos_s1_path),
            "cross-spec resurrection: spec-2 must not see any spec-1 delete"
        );
    }

    /// FK2.3 bar: seq-sorted lists + `partition_point` preserve the pre-FK2.3 applicable set
    /// under multi-spec + multi-sequence. Spec-1 deletes at seq {3,5,7} and spec-2 at {4,6};
    /// a data file at seq 4 under each spec must see the correct tail of ITS OWN list only.
    ///
    /// MUTATION: off-by-one in equality (`>=` instead of `>`) or position (`>` instead of `>=`)
    /// shifts a boundary delete into/out of the set (RED). Keying by Struct alone mixes both
    /// specs' tails (RED).
    #[test]
    fn test_fk2_3_multi_spec_seq_sorted_partition_point_identical_sets() {
        let partition = Struct::from_iter([Some(Literal::long(42))]);

        // Spec 1: eq@3, eq@5, eq@7, pos@3, pos@5, pos@7
        let s1_eq: Vec<DataFile> = [3i64, 5, 7]
            .into_iter()
            .map(|_| build_partitioned_eq_delete(&partition, 1))
            .collect();
        let s1_pos: Vec<DataFile> = [3i64, 5, 7]
            .into_iter()
            .map(|_| build_partitioned_pos_delete(&partition, 1))
            .collect();
        // Spec 2: eq@4, eq@6, pos@4, pos@6
        let s2_eq: Vec<DataFile> = [4i64, 6]
            .into_iter()
            .map(|_| build_partitioned_eq_delete(&partition, 2))
            .collect();
        let s2_pos: Vec<DataFile> = [4i64, 6]
            .into_iter()
            .map(|_| build_partitioned_pos_delete(&partition, 2))
            .collect();

        let mut contexts = Vec::new();
        for (seq, file) in [3i64, 5, 7].into_iter().zip(s1_eq.iter()) {
            contexts.push(DeleteFileContext {
                manifest_entry: build_added_manifest_entry(seq, file).into(),
                partition_spec_id: 1,
            });
        }
        for (seq, file) in [3i64, 5, 7].into_iter().zip(s1_pos.iter()) {
            contexts.push(DeleteFileContext {
                manifest_entry: build_added_manifest_entry(seq, file).into(),
                partition_spec_id: 1,
            });
        }
        for (seq, file) in [4i64, 6].into_iter().zip(s2_eq.iter()) {
            contexts.push(DeleteFileContext {
                manifest_entry: build_added_manifest_entry(seq, file).into(),
                partition_spec_id: 2,
            });
        }
        for (seq, file) in [4i64, 6].into_iter().zip(s2_pos.iter()) {
            contexts.push(DeleteFileContext {
                manifest_entry: build_added_manifest_entry(seq, file).into(),
                partition_spec_id: 2,
            });
        }
        // Shuffle insertion order so a missing sort-at-build would expose order-dependent bugs
        // if the partition_point assumed sorted input that wasn't sorted.
        contexts.reverse();

        let index = PopulatedDeleteFileIndex::new(contexts);
        let data_s1 = build_partitioned_data_file(&partition, 1);
        let data_s2 = build_partitioned_data_file(&partition, 2);

        // data_seq = 4:
        //   s1 eq (>4): eq@5, eq@7 ; s1 pos (>=4): pos@5, pos@7  (pos@3 drops)
        //   s2 eq (>4): eq@6       ; s2 pos (>=4): pos@4, pos@6
        let paths_s1: Vec<String> = index
            .get_deletes_for_data_file(&data_s1, Some(4))
            .expect("s1")
            .into_iter()
            .map(|d| d.file_path)
            .collect();
        let paths_s2: Vec<String> = index
            .get_deletes_for_data_file(&data_s2, Some(4))
            .expect("s2")
            .into_iter()
            .map(|d| d.file_path)
            .collect();

        assert_eq!(
            paths_s1,
            vec![
                s1_eq[1].file_path().to_string(),  // eq@5
                s1_eq[2].file_path().to_string(),  // eq@7
                s1_pos[1].file_path().to_string(), // pos@5
                s1_pos[2].file_path().to_string(), // pos@7
            ],
            "spec-1 seq-4 applicable tail must be eq{{5,7}} + pos{{5,7}}"
        );
        assert_eq!(
            paths_s2,
            vec![
                s2_eq[1].file_path().to_string(),  // eq@6
                s2_pos[0].file_path().to_string(), // pos@4
                s2_pos[1].file_path().to_string(), // pos@6
            ],
            "spec-2 seq-4 applicable tail must be eq{{6}} + pos{{4,6}}"
        );
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
