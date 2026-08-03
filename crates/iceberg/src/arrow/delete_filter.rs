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

use arrow_arith::boolean::and;
use arrow_array::{Array, BooleanArray, RecordBatch};
use arrow_select::filter::filter_record_batch;
use tokio::sync::Notify;
use tokio::sync::futures::OwnedNotified;
use tokio::sync::oneshot::Receiver;

use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
use crate::arrow::equality_delete_set::EqDeleteKeySet;
use crate::arrow::record_batch_predicate::evaluate_predicate_to_mask;
use crate::delete_vector::DeleteVector;
use crate::expr::Predicate::AlwaysTrue;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::io::FileIO;
use crate::scan::{FileScanTask, FileScanTaskDeleteFile};
use crate::spec::DataContentType;
use crate::{Error, ErrorKind, Result};

#[derive(Debug)]
enum EqDelState {
    Loading(Arc<Notify>),
    /// The resolved equality-delete file: its authoritative survival [`Predicate`] (always present —
    /// the oracle and the fallback) and, when every key column is type-eligible, the hashed
    /// [`EqDeleteKeySet`] accelerator for the O(R) apply fast path.
    Loaded(Predicate, Option<EqDeleteKeySet>),
    /// The load failed terminally: the loader dropped the oneshot sender without ever sending a
    /// predicate. This happens on ANY error or cancellation in the load → parse → send window
    /// (malformed `equality_ids`, an unreadable delete file, a schema-evolution or parse failure,
    /// or the whole load stream being torn down because a sibling task errored). Waiters MUST treat
    /// this as terminal — surfacing absence so the caller raises a typed error — instead of
    /// re-waiting on the notifier, which would block the scan forever.
    Failed,
}

/// State tracking for positional delete files.
/// Unlike equality deletes, positional deletes must be fully loaded before
/// the ArrowReader proceeds because retrieval is synchronous and non-blocking.
#[derive(Debug)]
enum PosDelState {
    /// The file is currently being loaded by a task.
    /// The notifier allows other tasks to wait for completion.
    Loading(Arc<Notify>),
    /// The file has been fully loaded and merged into the delete vector map.
    Loaded,
    /// The load failed terminally: the task that claimed this file (see
    /// [`DeleteFilter::try_start_pos_del_load`]) died without ever publishing its delete vectors.
    /// This happens on ANY error or cancellation in the claim → read → parse → merge window (an
    /// unreadable or corrupt delete file, a negative position, the whole load stream being torn
    /// down because a sibling task errored, a panic, or a runtime shutdown). The claiming task is
    /// the sole writer for its file and runs once, so the state can never advance on its own:
    /// waiters MUST treat this as terminal and surface a typed error instead of re-waiting on a
    /// notifier that has already fired for the last time — which would block the scan forever.
    /// Mirrors [`EqDelState::Failed`] (and `DeleteFileIndexState::Failed` in
    /// [`crate::delete_file_index`]).
    ///
    /// Carries the CAUSE, rendered into every waiter's (and every later claimant's) error — the
    /// error itself cannot be carried, since it reaches only the task that produced it and
    /// [`Error`] is not `Clone`. The claiming task records it with
    /// [`PosDelLoadGuard::note_failure`] on the paths where it has one; the paths where it does not
    /// (an unwind, a cancelled future, a runtime shutdown) publish a generic reason. Deliberately
    /// asymmetric with [`EqDelState::Failed`]: there the terminal transition is made by the
    /// publisher task, which observes only a dropped oneshot sender and so has no cause to record.
    Failed(String),
}

/// The memo key for one task-shaped positional-delete resolution: the task's data file path plus
/// the SORTED, DEDUPLICATED claim keys of its positional delete sources. Two tasks (or two loads of
/// one task) with the same data file and the same delete set resolve to the same key and share one
/// merged vector — the shared-state analogue of Java's per-task
/// `DeleteFilter.deleteRowPositions` memo field (`DeleteFilter.deletedRowPositions()`, 1.10.0
/// bytecode offsets 0-4: return the cached index when non-null).
type PosDelResolutionKey = (String, Vec<String>);

#[derive(Debug, Default)]
struct DeleteFileFilterState {
    /// Parsed positional-delete content, PER SOURCE — the load cache. Keyed by the source's claim
    /// key ([`pos_del_claim_key`]: the parquet delete file's path, or `{puffin path}@{offset}` for
    /// a deletion-vector blob); each value maps a DATA file path to the positions that source
    /// deletes from it. This is Java's cache shape exactly: `BaseDeleteLoader.getOrReadPosDeletes`
    /// caches `readPosDeletes(deleteFile)` — a `CharSequenceMap<PositionDeleteIndex>` keyed by data
    /// file — under `deleteFile.location()` (1.10.0 bytecode offsets 22-39).
    ///
    /// Keeping the per-source maps SEPARATE (instead of merging them into one shared
    /// data-file-keyed map at load time, as this state did before) is what scopes delete
    /// APPLICATION to each task's own delete set: a source loaded for one task can no longer
    /// contribute deletions to a task that does not list it (the R117 cross-task over-delete).
    /// An installed map is never mutated again (each claim key is loaded exactly once — the
    /// [`PosDelState`] machinery is the single-writer guarantee), so resolution can snapshot the
    /// `Arc`s and union outside the lock.
    pos_del_contributions: HashMap<String, Arc<HashMap<String, DeleteVector>>>,
    /// Memoized per-task merged vectors — see [`PosDelResolutionKey`]. Entries are only installed
    /// once every claim key they depend on is present in `pos_del_contributions`, and contributions
    /// are immutable once installed, so a memoized union can never go stale.
    ///
    /// Frozen as [`Arc<DeleteVector>`] (not `Arc<Mutex<…>>`): audit of the load → install →
    /// resolve path shows no post-publish mutation of a memoized vector — only reads
    /// (`contains` / `iter` / `is_empty` / range-walk keep-masks). See FK3 scout #12.
    resolved_pos_dels: HashMap<PosDelResolutionKey, Arc<DeleteVector>>,
    equality_deletes: HashMap<String, EqDelState>,
    positional_deletes: HashMap<String, PosDelState>,
}

/// The resolved merge-on-read deletes for a scan — position deletes, deletion vectors, and equality
/// deletes — plus the logic to apply them to Arrow batches.
///
/// This is the engine-facing analogue of Java `org.apache.iceberg.data.DeleteFilter`: a downstream
/// query engine that builds its OWN physical scan (its own Parquet read / `ExecutionPlan`) uses it to
/// REUSE Iceberg's delete resolution instead of reimplementing it (and its sequence-number,
/// DV-supersedes-position, and null-coercion rules). The typical loop, per [`FileScanTask`] obtained
/// from [`TableScan::plan_files`](crate::scan::TableScan::plan_files):
///
/// ```ignore
/// let deletes = DeleteFilter::load(&task, file_io).await?;
/// let eq_predicate = deletes.equality_delete_predicate(&task).await?;
/// let mut row_base = 0u64;
/// for batch in your_own_data_file_reader {     // batches of `task`'s data file, in file order
///     let n = batch.num_rows() as u64;
///     let surviving = deletes.apply(&task, batch, row_base, eq_predicate.as_ref())?;
///     row_base += n;
///     emit(surviving);
/// }
/// ```
///
/// A columnar engine that prefers to fold deletes into its own pushdown can instead read
/// [`deleted_row_positions`](Self::deleted_row_positions) (the position bitmap, ≈ Java
/// `deletedRowPositions()`) and [`equality_delete_predicate`](Self::equality_delete_predicate)
/// (≈ Java `eqDeletedRowFilter()`) directly and skip [`apply`](Self::apply).
#[derive(Clone, Debug, Default)]
pub struct DeleteFilter {
    state: Arc<RwLock<DeleteFileFilterState>>,
}

/// Action to take when trying to start loading a positional delete file
#[derive(Debug)]
pub(crate) enum PosDelLoadAction {
    /// The file is not loaded, the caller should load it. The guard carries the claim: publish it
    /// with [`PosDelLoadGuard::publish_loaded`] once the delete vectors are merged into the
    /// filter, or let it drop and every waiter gets a typed error instead of hanging.
    Load(PosDelLoadGuard),
    /// The file is already loaded, nothing to do.
    AlreadyLoaded,
    /// The file is currently being loaded by another task. The caller *must* wait — pass this
    /// future to [`DeleteFilter::wait_for_pos_del_load`] — to ensure data availability before
    /// returning, as subsequent access (`get_delete_vector`) is synchronous.
    ///
    /// The future is ARMED HERE, under the state lock, and handed to the caller already created.
    /// That is load-bearing: [`Notify::notify_waiters`] stores no permit and only wakes `Notified`
    /// futures that already EXIST when it fires, and `Notify::notified_owned` snapshots the
    /// notifier's `notify_waiters` counter at CALL time. Returning a bare `Arc<Notify>` for the
    /// caller to `.notified()` at the await site left a window between releasing the lock and
    /// creating the future in which the loader could publish + notify — the wakeup was then
    /// dropped and the waiting scan parked forever (upstream apache/iceberg-rust#2859, the same
    /// class as #2696 on the delete-file-index wait path). Creating the future while the lock is
    /// held closes it: the loader cannot notify until it has taken the WRITE lock, which cannot be
    /// granted while this lock is held, so any notification necessarily follows this arming.
    WaitFor(OwnedNotified),
}

/// Publishes the TERMINAL state of one positional-delete file's load and wakes its waiters.
///
/// Handed to the claiming task by [`DeleteFilter::try_start_pos_del_load`] — i.e. armed in the
/// same critical section that installs [`PosDelState::Loading`], so there is no window in which
/// the claim exists without its guard. [`PosDelLoadGuard::publish_loaded`] disarms it on the
/// success path; if that call is never reached, `Drop` publishes [`PosDelState::Failed`] instead,
/// so every waiter reaches a terminal state and gets a typed error rather than hanging forever.
/// `Drop` therefore covers every way the loading task can die without publishing: an early `?`
/// return (unreadable file, corrupt rows), a sibling task's error tearing down the shared load
/// stream, an unwind, and a runtime shutdown that drops the task's future.
///
/// Both paths write the state under the write lock and fire the notifier only AFTER releasing it,
/// so a woken waiter always observes the terminal state — the other half of the handshake with
/// [`PosDelLoadAction::WaitFor`]'s arming.
pub(crate) struct PosDelLoadGuard {
    state: Arc<RwLock<DeleteFileFilterState>>,
    notify: Arc<Notify>,
    file_path: String,
    armed: bool,
    /// The cause recorded by [`PosDelLoadGuard::note_failure`], published into
    /// [`PosDelState::Failed`] so waiters learn WHY the load died, not just THAT it did.
    failure_reason: Option<String>,
}

/// Renders the claim, not the whole guarded filter state: a `Debug` that reached for the state
/// would `try_read` a lock this guard's own `publish` may be holding.
impl std::fmt::Debug for PosDelLoadGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PosDelLoadGuard")
            .field("file_path", &self.file_path)
            .field("armed", &self.armed)
            .field("failure_reason", &self.failure_reason)
            .finish_non_exhaustive()
    }
}

impl PosDelLoadGuard {
    fn publish(&mut self, terminal: PosDelState) {
        {
            // Recover a poisoned guard rather than cascading the panic: this task is the sole
            // writer for its file and runs once, so recovering and completing the transition is
            // always the right move — a stranded `Loading` would hang every waiter below.
            let mut state = recover_poison(self.state.write());
            state
                .positional_deletes
                .insert(self.file_path.clone(), terminal);
        }
        self.armed = false;
        self.notify.notify_waiters();
    }

    /// Mark this positional delete file fully loaded and wake every waiter. Call it only AFTER the
    /// file's delete vectors have been merged into the filter: a woken waiter reads them
    /// synchronously and would otherwise see an empty or partial result.
    pub(crate) fn publish_loaded(mut self) {
        self.publish(PosDelState::Loaded);
    }

    /// Record the error that is about to end this load, then hand it back for `?` propagation.
    ///
    /// Only the task holding the guard ever sees that error; every OTHER consumer of this file —
    /// a concurrent waiter, and any later claimant on the same (result-caching) loader — reaches
    /// the terminal state instead, so without this the cause is lost to them and they learn only
    /// THAT the load died. Use it on every failure path that has a cause in hand; the paths that
    /// do not (an unwind, a future dropped by a runtime teardown, a sibling task's error tearing
    /// the shared load stream down) still publish `Failed` from `Drop`, with a generic reason.
    pub(crate) fn note_failure(&mut self, error: Error) -> Error {
        self.failure_reason = Some(error.to_string());
        error
    }
}

impl Drop for PosDelLoadGuard {
    fn drop(&mut self) {
        if self.armed {
            let reason = self.failure_reason.take().unwrap_or_else(|| {
                "no cause was recorded — the load was cancelled, panicked, or the runtime was \
                 shut down"
                    .to_string()
            });
            self.publish(PosDelState::Failed(reason));
        }
    }
}

/// Publishes the TERMINAL state of one equality-delete file's load and wakes its waiters.
///
/// The equality-delete counterpart of [`PosDelLoadGuard`], armed by
/// [`DeleteFilter::try_start_eq_del_load`] in the same critical section that installs
/// [`EqDelState::Loading`]. It carries the notifier that claim installed, so the waiter-visible
/// notifier and the one that eventually fires are THE SAME object — minting a fresh notifier at
/// publish-registration time would strand any waiter that armed on the claim's notifier in
/// between.
///
/// [`EqDelLoadGuard::spawn_publisher`] hands the guard to the task that awaits the loader's
/// oneshot; the guard is CAPTURED by that task's future rather than constructed inside it, so a
/// future dropped before it is ever polled (a runtime torn down between `spawn` and the first
/// poll) still runs `Drop` and publishes [`EqDelState::Failed`].
pub(crate) struct EqDelLoadGuard {
    state: Arc<RwLock<DeleteFileFilterState>>,
    notify: Arc<Notify>,
    file_path: String,
    armed: bool,
}

impl EqDelLoadGuard {
    fn publish(&mut self, terminal: EqDelState) {
        {
            let mut state = recover_poison(self.state.write());
            state
                .equality_deletes
                .insert(self.file_path.clone(), terminal);
        }
        self.armed = false;
        self.notify.notify_waiters();
    }

    /// Spawn the task that turns the loader's oneshot into this file's terminal state.
    ///
    /// The loader sends the parsed predicate (and optional key set) once the eq-delete file is
    /// read. If the SENDER is instead dropped without sending — which happens on ANY error or
    /// cancellation in the load → parse → send window (a malformed `equality_ids`, an unreadable
    /// delete file, a schema-evolution or parse failure, or the whole load stream being torn down
    /// because a sibling task errored) — `recv` errs and the entry moves to the terminal
    /// [`EqDelState::Failed`], STILL waking the waiters: leaving it `Loading` strands every
    /// predicate / key-set waiter on the notifier forever. The waiters read `Failed` as absence
    /// and surface a typed error to the caller.
    pub(crate) fn spawn_publisher(mut self, eq_del: Receiver<(Predicate, Option<EqDeleteKeySet>)>) {
        crate::runtime::spawn(async move {
            let terminal = match eq_del.await {
                Ok((predicate, key_set)) => EqDelState::Loaded(predicate, key_set),
                Err(_) => EqDelState::Failed,
            };
            self.publish(terminal);
        });
    }
}

impl Drop for EqDelLoadGuard {
    fn drop(&mut self) {
        if self.armed {
            self.publish(EqDelState::Failed);
        }
    }
}

/// The outcome of consulting one equality-delete entry once — see
/// [`DeleteFilter::lookup_or_arm_eq_del`].
enum EqDelLookup<T> {
    /// The entry had reached a terminal state (or is unknown); this is the answer.
    Ready(Option<T>),
    /// The entry was still loading. Await this ALREADY-ARMED future, then read the state again.
    Wait(OwnedNotified),
}

/// Recover a poisoned lock guard instead of cascading the panic to every subsequent scan.
///
/// The guarded [`DeleteFileFilterState`] is a set of `HashMap`s whose critical sections perform
/// only `insert`/`get`/`clone` — no re-entrant user code that could tear a collection
/// mid-mutation — so a guard left behind by a panicked holder still wraps a structurally coherent
/// state. Recovering it via [`std::sync::PoisonError::into_inner`] keeps concurrent scans alive
/// rather than turning one thread's panic into a poison-panic in every reader/writer. This is the
/// crate's established policy for these delete-path locks (see `arrow/reader.rs`). Memoized
/// positional delete vectors are frozen as [`Arc<DeleteVector>`] and are not themselves locked.
fn recover_poison<G>(result: std::sync::LockResult<G>) -> G {
    result.unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The claim/cache key under which one positional-delete SOURCE is loaded and its parsed
/// contribution installed — the single source of truth shared by the loader (claim + install)
/// and by [`DeleteFilter::resolve_delete_vector`] (application), so the two sides can never
/// drift apart and silently drop deletes.
///
/// * A parquet position-delete file is keyed by its own path.
/// * A deletion vector (a position delete in PUFFIN format) is keyed `{puffin path}@{offset}` —
///   one Puffin file holds many DV blobs, so the bare file path would collide them.
/// * Anything else (equality deletes, data files) has no positional claim key: `None`.
///
/// Returns `None` for a Puffin entry with a missing or negative `content_offset` — invalid
/// metadata that `CachingDeleteFileLoader::validate_deletion_vector_task` fails loud on at load
/// time, so such an entry can never have an installed contribution to resolve.
pub(crate) fn pos_del_claim_key(delete: &FileScanTaskDeleteFile) -> Option<String> {
    if delete.file_type != DataContentType::PositionDeletes {
        return None;
    }
    if delete.file_format == crate::spec::DataFileFormat::Puffin {
        let offset = delete.content_offset.filter(|offset| *offset >= 0)?;
        Some(format!("{}@{offset}", delete.file_path))
    } else {
        Some(delete.file_path.clone())
    }
}

impl DeleteFilter {
    /// Retrieve the merged positional-delete vector for a file scan task — the union of the
    /// contributions that the task's OWN delete files make to the task's data file. See
    /// [`Self::resolve_delete_vector`].
    pub(crate) fn get_delete_vector(
        &self,
        file_scan_task: &FileScanTask,
    ) -> Option<Arc<DeleteVector>> {
        self.resolve_delete_vector(&file_scan_task.deletes, file_scan_task.data_file_path())
    }

    /// Resolve the positional deletes that `deletes` (a task's delete files) apply to
    /// `data_file_path` (that task's data file): the union, over the task's own positional
    /// sources only, of each source's contribution to this data file.
    ///
    /// This is Java's per-task scope exactly: Java builds one `data.DeleteFilter` per task over
    /// `task.deletes()` alone (constructor bytecode offsets 51-208 partition the GIVEN list), and
    /// `deletedRowPositions()` merges `deleteLoader().loadPositionDeletes(this.posDeletes,
    /// this.filePath)` (offsets 19-37) — per delete file, the cached contribution map is consulted
    /// with `getOrDefault(filePath, PositionDeleteIndex.empty())` (`BaseDeleteLoader
    /// .getOrReadPosDeletes`, offsets 41-50) and the per-file indexes merged into a FRESH index
    /// (`PositionDeleteIndexUtil.merge`, offsets 0-26). A delete file loaded for a DIFFERENT task
    /// therefore never contributes here, and a listed delete file's rows that name OTHER data
    /// files are ignored for this one.
    ///
    /// The merged vector is memoized per [`PosDelResolutionKey`] (Java memoizes the merge in the
    /// per-task `deleteRowPositions` field), so repeated resolution of one task — and every other
    /// task with the identical delete set + data file — returns the SAME `Arc`.
    ///
    /// Returns `None` when no listed source contributes any position for this data file. Two
    /// unreachable-by-contract shapes also resolve as `None`, loudly: a listed positional source
    /// with an underivable claim key, and one whose contribution was never installed (resolution
    /// before `load_deletes` completed — the loader awaits every listed source before handing the
    /// filter out, and a failed load fails the whole scan instead). Both log at WARN because
    /// silently dropping a REAL source here would resurrect deleted rows.
    pub(crate) fn resolve_delete_vector(
        &self,
        deletes: &[FileScanTaskDeleteFile],
        data_file_path: &str,
    ) -> Option<Arc<DeleteVector>> {
        let mut claim_keys: Vec<String> = Vec::new();
        for delete in deletes {
            if delete.file_type != DataContentType::PositionDeletes {
                continue;
            }
            match pos_del_claim_key(delete) {
                Some(key) => claim_keys.push(key),
                None => {
                    // Invalid DV metadata (missing/negative offset) — the loader fails loud on
                    // this shape before any contribution exists, so a filter being resolved can
                    // only see it through contract misuse.
                    tracing::warn!(
                        delete_file = %delete.file_path,
                        "skipping a positional delete source with an underivable claim key \
                         (invalid deletion-vector metadata); its load would have failed the scan"
                    );
                }
            }
        }
        claim_keys.sort_unstable();
        claim_keys.dedup();
        if claim_keys.is_empty() {
            return None;
        }

        // Snapshot the contribution maps under the read lock; union OUTSIDE it (contributions are
        // immutable once installed, so the snapshot cannot go stale). Poison is recovered rather
        // than swallowed as `None` — a poison-induced `None` is read as "no positional deletes"
        // and would silently resurrect deleted rows (the `recover_poison` policy of this file).
        let mut contributions: Vec<Arc<HashMap<String, DeleteVector>>> =
            Vec::with_capacity(claim_keys.len());
        let mut every_source_installed = true;
        {
            let state = recover_poison(self.state.read());
            if let Some(resolved) = state
                .resolved_pos_dels
                .get(&(data_file_path.to_string(), claim_keys.clone()))
            {
                return Some(resolved.clone());
            }
            for key in &claim_keys {
                match state.pos_del_contributions.get(key) {
                    Some(contribution) => contributions.push(contribution.clone()),
                    None => {
                        // Reachable only by resolving before this source's load completed (the
                        // loader publishes every listed source before delivering the filter).
                        // Never memoize a union computed without every listed source.
                        every_source_installed = false;
                        tracing::warn!(
                            claim_key = %key,
                            data_file = %data_file_path,
                            "resolving positional deletes for a source whose contribution is not \
                             installed; load_deletes for this task has not completed"
                        );
                    }
                }
            }
        }

        // OR-by-reference (no roaring clone of each contribution): each contribution is frozen
        // after install, and the memoized merge is published as `Arc<DeleteVector>` once.
        let mut merged: Option<DeleteVector> = None;
        for contribution in &contributions {
            if let Some(vector) = contribution.get(data_file_path) {
                merged
                    .get_or_insert_with(DeleteVector::default)
                    .merge(vector);
            }
        }
        let merged = Arc::new(merged?);

        if every_source_installed {
            // Double-checked install: a concurrent resolver may have memoized the same key while
            // the union above ran outside the lock — return THEIRS so every resolver of one task
            // shape shares a single frozen Arc.
            let mut state = recover_poison(self.state.write());
            let entry = state
                .resolved_pos_dels
                .entry((data_file_path.to_string(), claim_keys))
                .or_insert_with(|| merged.clone());
            return Some(entry.clone());
        }
        Some(merged)
    }

    /// Attempts to claim an equality delete file for loading, returning the guard that publishes
    /// its terminal state. `None` means another task already owns it (or it already reached a
    /// terminal state) and this caller must not load it.
    pub(crate) fn try_start_eq_del_load(&self, file_path: &str) -> Option<EqDelLoadGuard> {
        let mut state = recover_poison(self.state.write());

        // Skip if already loaded/loading/failed - another task owns it. A terminal `Failed` is NOT
        // re-claimed: it is cached for the lifetime of this filter exactly as `Loaded` is, so the
        // waiters' post-wake re-read stays unambiguous (a re-claim could install a fresh `Loading`
        // under a woken waiter). The waiter reads `Failed` as absence and the caller raises a
        // typed error naming the file.
        if state.equality_deletes.contains_key(file_path) {
            return None;
        }

        // Mark as loading to prevent duplicate work. The guard carries THIS notifier, so the
        // notifier waiters arm on is the notifier that eventually fires.
        let notify = Arc::new(Notify::new());
        state
            .equality_deletes
            .insert(file_path.to_string(), EqDelState::Loading(notify.clone()));

        Some(EqDelLoadGuard {
            state: self.state.clone(),
            notify,
            file_path: file_path.to_string(),
            armed: true,
        })
    }

    /// Attempts to mark a positional delete file as "loading".
    ///
    /// Returns an action dictating whether the caller should load the file (carrying the guard
    /// that publishes the outcome), wait for another task to load it, or do nothing.
    ///
    /// Errs when a previous loader for this file terminated without publishing
    /// ([`PosDelState::Failed`]): the state is terminal, so a fresh claim would be a lie and a
    /// wait would never end. The scan fails loudly instead — never silently without this file's
    /// deletes, which would resurrect deleted rows.
    pub(crate) fn try_start_pos_del_load(&self, file_path: &str) -> Result<PosDelLoadAction> {
        let mut state = recover_poison(self.state.write());

        if let Some(existing) = state.positional_deletes.get(file_path) {
            match existing {
                PosDelState::Loaded => return Ok(PosDelLoadAction::AlreadyLoaded),
                // ARM HERE, under the lock — see `PosDelLoadAction::WaitFor`.
                PosDelState::Loading(notify) => {
                    return Ok(PosDelLoadAction::WaitFor(notify.clone().notified_owned()));
                }
                PosDelState::Failed(reason) => {
                    return Err(pos_del_load_failed_error(file_path, reason));
                }
            }
        }

        let notify = Arc::new(Notify::new());
        state
            .positional_deletes
            .insert(file_path.to_string(), PosDelState::Loading(notify.clone()));

        Ok(PosDelLoadAction::Load(PosDelLoadGuard {
            state: self.state.clone(),
            notify,
            file_path: file_path.to_string(),
            armed: true,
            failure_reason: None,
        }))
    }

    /// Wait for another task's in-flight positional-delete load to reach a terminal state.
    ///
    /// `notified` MUST be the future armed by [`Self::try_start_pos_del_load`] under the state
    /// lock; awaiting a `Notified` created here instead would reopen the lost-wakeup window.
    /// Returns once the file's delete vectors are merged into the filter, or a typed error if the
    /// loading task died without publishing them — never a hang.
    pub(crate) async fn wait_for_pos_del_load(
        &self,
        file_path: &str,
        notified: OwnedNotified,
    ) -> Result<()> {
        notified.await;

        // The loading task publishes a TERMINAL state under the write lock and only then fires the
        // notifier, so a woken waiter always observes `Loaded` or `Failed`. Neither is ever
        // replaced (`try_start_pos_del_load` re-claims neither), so anything else here means the
        // notifier fired without a terminal transition — surface it rather than re-waiting.
        match recover_poison(self.state.read())
            .positional_deletes
            .get(file_path)
        {
            Some(PosDelState::Loaded) => Ok(()),
            Some(PosDelState::Failed(reason)) => Err(pos_del_load_failed_error(file_path, reason)),
            _ => Err(Error::new(
                ErrorKind::Unexpected,
                format!(
                    "the positional delete file '{file_path}' notified its waiters without \
                     reaching a terminal load state"
                ),
            )),
        }
    }

    /// Read one equality-delete entry once: answer outright if it is terminal, otherwise ARM the
    /// notifier.
    ///
    /// The arming MUST happen here, while the read lock is still held — the same handshake
    /// [`PosDelLoadAction::WaitFor`] documents. [`Notify::notify_waiters`] stores no permit and
    /// only wakes `Notified` futures that already EXIST when it fires, and
    /// `Notify::notified_owned` snapshots the notifier's `notify_waiters` counter at CALL time,
    /// so a future created after the loader published is never woken. Cloning the `Arc<Notify>`
    /// out and calling `.notified()` at the await site left exactly that window open between
    /// releasing the read lock and creating the future: a load that published + notified in it
    /// dropped the wakeup and the querying scan awaited forever (upstream
    /// apache/iceberg-rust#2859). Creating the future under the read lock closes it — the
    /// publisher cannot notify until it has taken the WRITE lock, which cannot be granted while
    /// this read lock is held.
    ///
    /// `Ready(None)` covers both an unknown file and a terminally [`EqDelState::Failed`] one: the
    /// caller surfaces absence, and `build_equality_delete_predicate` turns it into a typed error
    /// rather than blocking forever on a notifier that already fired.
    fn lookup_or_arm_eq_del<T>(
        &self,
        file_path: &str,
        project: impl FnOnce(&Predicate, &Option<EqDeleteKeySet>) -> T,
    ) -> EqDelLookup<T> {
        match recover_poison(self.state.read())
            .equality_deletes
            .get(file_path)
        {
            None | Some(EqDelState::Failed) => EqDelLookup::Ready(None),
            Some(EqDelState::Loaded(predicate, key_set)) => {
                EqDelLookup::Ready(Some(project(predicate, key_set)))
            }
            Some(EqDelState::Loading(notify)) => EqDelLookup::Wait(notify.clone().notified_owned()),
        }
    }

    /// Retrieve the equality delete predicate for a given eq delete file path
    pub(crate) async fn get_equality_delete_predicate_for_delete_file_path(
        &self,
        file_path: &str,
    ) -> Option<Predicate> {
        match self.lookup_or_arm_eq_del(file_path, |predicate, _| predicate.clone()) {
            EqDelLookup::Ready(predicate) => return predicate,
            EqDelLookup::Wait(notified) => notified.await,
        }

        // Once the notifier fires the entry is terminal: `Loaded` on success, `Failed` on a load
        // error, and neither is ever replaced (`try_start_eq_del_load` never re-claims a present
        // entry). Treat anything other than `Loaded` (Failed, or — defensively — a still-Loading
        // or absent entry) as absence so the caller surfaces a typed error instead of re-waiting.
        match self.lookup_or_arm_eq_del(file_path, |predicate, _| predicate.clone()) {
            EqDelLookup::Ready(predicate) => predicate,
            EqDelLookup::Wait(_) => None,
        }
    }

    /// Retrieve the hashed [`EqDeleteKeySet`] accelerator for an eq-delete file, awaiting its load.
    /// `Some(set)` means the file is fast-path-eligible (all key columns are type-eligible);
    /// `Some(None)`-style absence is folded into `None` here — the caller then uses the predicate
    /// path. Returns `None` if the file is unknown.
    pub(crate) async fn get_equality_delete_keyset_for_delete_file_path(
        &self,
        file_path: &str,
    ) -> Option<EqDeleteKeySet> {
        // A terminally-failed (or unknown) load surfaces as "no key set", routing this file's task
        // onto the predicate path — which then raises the typed error — instead of blocking. The
        // outer `Option` is the entry's presence, the inner one the file's fast-path eligibility.
        match self.lookup_or_arm_eq_del(file_path, |_, key_set| key_set.clone()) {
            EqDelLookup::Ready(key_set) => return key_set.flatten(),
            EqDelLookup::Wait(notified) => notified.await,
        }

        // As in `get_equality_delete_predicate_for_delete_file_path`: after the notifier fires the
        // entry is terminal; anything other than `Loaded` yields `None` (use the predicate path)
        // rather than re-waiting.
        match self.lookup_or_arm_eq_del(file_path, |_, key_set| key_set.clone()) {
            EqDelLookup::Ready(key_set) => key_set.flatten(),
            EqDelLookup::Wait(_) => None,
        }
    }

    /// Collect the hashed key sets for ALL of `task`'s equality-delete files — `Some(sets)` only if
    /// EVERY eq-delete file is fast-path-eligible and they share one key-column schema (so their
    /// per-file delete masks can be OR-combined under one tuple shape). Returns `None` (use the
    /// predicate path for the whole task) if the task has no eq-deletes, any file is ineligible, or
    /// the files disagree on key columns. This is the routing gate for the O(R) fast path.
    pub(crate) async fn collect_equality_delete_keysets(
        &self,
        task: &FileScanTask,
    ) -> Option<Vec<EqDeleteKeySet>> {
        let mut sets: Vec<EqDeleteKeySet> = Vec::new();
        let mut shared_key_ids: Option<Vec<i32>> = None;
        for delete in task.deletes.iter() {
            if !is_equality_delete(delete) {
                continue;
            }
            // Any eq-delete file without a key set (ineligible type) disables the fast path.
            let set = self
                .get_equality_delete_keyset_for_delete_file_path(&delete.file_path)
                .await?;
            match &shared_key_ids {
                None => shared_key_ids = Some(set.key_field_ids()),
                Some(ids) if *ids != set.key_field_ids() => return None,
                Some(_) => {}
            }
            sets.push(set);
        }
        if sets.is_empty() { None } else { Some(sets) }
    }

    /// Builds eq delete predicate for the provided task.
    pub(crate) async fn build_equality_delete_predicate(
        &self,
        file_scan_task: &FileScanTask,
    ) -> Result<Option<BoundPredicate>> {
        // * Filter the task's deletes into just the Equality deletes
        // * Retrieve the unbound predicate for each from self.state.equality_deletes
        // * Logical-AND them all together to get a single combined `Predicate`
        // * Bind the predicate to the task's schema to get a `BoundPredicate`

        let mut combined_predicate = AlwaysTrue;
        for delete in file_scan_task.deletes.iter() {
            if !is_equality_delete(delete) {
                continue;
            }

            let Some(predicate) = self
                .get_equality_delete_predicate_for_delete_file_path(&delete.file_path)
                .await
            else {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "Missing predicate for equality delete file '{}'",
                        delete.file_path
                    ),
                ));
            };

            combined_predicate = combined_predicate.and(predicate);
        }

        if combined_predicate == AlwaysTrue {
            return Ok(None);
        }

        let bound_predicate = combined_predicate
            .bind(file_scan_task.schema.clone(), file_scan_task.case_sensitive)?;
        Ok(Some(bound_predicate))
    }

    /// Install the parsed contribution map of ONE freshly loaded positional-delete source — the
    /// `data file path → positions` map its rows produced — under the claim the loading task holds.
    /// Call it BEFORE [`PosDelLoadGuard::publish_loaded`], in the same await-free block: a woken
    /// waiter resolves synchronously, so publishing first (or being cancelled in between) would
    /// hand it an absent contribution.
    ///
    /// Taking the [`PosDelLoadGuard`] rather than a bare key ties the install to the claim: only
    /// the single task that owns a source's load can install its contribution, and only under the
    /// exact key it claimed (key drift between claim, install and resolution would silently drop
    /// deletes). Installing over a present entry is structurally unreachable (the claim machinery
    /// hands out one `Load` per key per filter lifetime); `insert` keeps the operation total.
    pub(crate) fn install_pos_del_contribution(
        &self,
        claim: &PosDelLoadGuard,
        contribution: HashMap<String, DeleteVector>,
    ) {
        let mut state = recover_poison(self.state.write());
        state
            .pos_del_contributions
            .insert(claim.file_path.clone(), Arc::new(contribution));
    }
}

/// The typed error every consumer of a terminally-failed positional-delete load receives — the
/// claim-time and the post-wake paths must render the same failure, so it lives in one place.
///
/// `reason` is the cause carried by [`PosDelState::Failed`]: the failing task's own error where it
/// had one ([`PosDelLoadGuard::note_failure`]), else a generic reason. It is rendered inline rather
/// than attached with `with_source`, because it is a message the failing task left behind, not an
/// [`Error`] this one is wrapping — nothing is dropped from a source chain here.
fn pos_del_load_failed_error(file_path: &str, reason: &str) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        format!(
            "the loader for positional delete file '{file_path}' terminated without publishing \
             its deletes: {reason}"
        ),
    )
}

/// Engine-facing API — the stable public surface mirroring Java `org.apache.iceberg.data.DeleteFilter`.
impl DeleteFilter {
    /// Load and resolve every merge-on-read delete (position deletes, deletion vectors, and equality
    /// deletes) that applies to `task`, reading the delete files via `file_io`. Run this concurrently
    /// with your own data-file read (e.g. `tokio::join!`): position deletes and deletion vectors are
    /// fully resolved when this returns; equality-delete predicates resolve lazily on the first
    /// [`equality_delete_predicate`](Self::equality_delete_predicate) call. Hides the internal
    /// caching delete-file loader.
    pub async fn load(task: &FileScanTask, file_io: FileIO) -> Result<Self> {
        let loader = CachingDeleteFileLoader::new(file_io, task.deletes.len().max(1));
        loader
            .load_deletes(&task.deletes, task.schema_ref())
            .await
            .map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    "the delete-file loader was dropped before delivering the delete filter",
                )
                .with_source(e)
            })?
    }

    /// The positional deletes that apply to `task`'s data file — the bitmap of deleted 0-based file
    /// positions (parquet position deletes and/or a deletion vector, already merged) — or `None`.
    /// Mirrors Java `DeleteFilter.deletedRowPositions()`. Synchronous: fully populated once
    /// [`load`](Self::load) returns.
    pub fn deleted_row_positions(&self, task: &FileScanTask) -> Option<Arc<DeleteVector>> {
        self.get_delete_vector(task)
    }

    /// The combined equality-delete predicate for `task`, bound to the task schema — a row SURVIVES
    /// iff it evaluates TRUE (the predicate is the negation of the delete condition). `None` when the
    /// task has no equality deletes. Mirrors Java `DeleteFilter.eqDeletedRowFilter()`.
    pub async fn equality_delete_predicate(
        &self,
        task: &FileScanTask,
    ) -> Result<Option<BoundPredicate>> {
        self.build_equality_delete_predicate(task).await
    }

    /// Apply `task`'s deletes to one Arrow `batch` of its data file, returning the surviving rows.
    ///
    /// `row_base` is the absolute 0-based position of `batch`'s first row within the data file — i.e.
    /// the `_pos` of row 0 (see
    /// [`RESERVED_COL_NAME_POS`](crate::metadata_columns::RESERVED_COL_NAME_POS)). Batches MUST be
    /// supplied in file order with no rows skipped, so positions stay aligned. `equality_predicate` is
    /// the once-resolved result of [`equality_delete_predicate`](Self::equality_delete_predicate);
    /// pass `None` if the task has no equality deletes. To apply equality deletes, `batch` must carry
    /// the equality-delete columns (resolved by Iceberg field id).
    ///
    /// Mirrors Java `DeleteFilter.filter(...)`: combines the positional keep-mask
    /// (`!deleted(row_base + i)`) with the equality/​residual predicate mask (NULLs coerced to `false`,
    /// matching the Parquet `RowFilter`) and filters the batch. This is the public counterpart of the
    /// reader's internal `survival_mask`.
    pub fn apply(
        &self,
        task: &FileScanTask,
        batch: RecordBatch,
        row_base: u64,
        equality_predicate: Option<&BoundPredicate>,
    ) -> Result<RecordBatch> {
        let num_rows = batch.num_rows();

        // Positional deletes → a keep-mask of `!deleted` over [row_base, row_base + num_rows).
        // The memoized vector is frozen (`Arc<DeleteVector>`); apply is lock-free on the bitmap.
        let positional_mask: Option<BooleanArray> = match self.get_delete_vector(task) {
            Some(deletes) => {
                if deletes.is_empty() {
                    None
                } else {
                    // Range-walk the delete window — byte-identical to the per-row `!contains` probe,
                    // O(D_window) instead of O(num_rows). See `positional_delete_keep_mask`.
                    Some(positional_delete_keep_mask(
                        deletes.as_ref(),
                        row_base,
                        num_rows,
                    ))
                }
            }
            None => None,
        };

        // Equality-delete predicate → a keep-mask (true ⇒ survives). The mask is already
        // two-valued under Java nulls-first semantics (a NULL key cell survives a value delete,
        // matching Java's StructLikeSet equality); the coercion is defense in depth.
        let predicate_mask: Option<BooleanArray> = match equality_predicate {
            Some(predicate) => Some(coerce_nulls_to_false(&evaluate_predicate_to_mask(
                predicate, &batch,
            )?)),
            None => None,
        };

        let mask = match (positional_mask, predicate_mask) {
            (None, None) => return Ok(batch),
            (Some(mask), None) | (None, Some(mask)) => mask,
            (Some(pos), Some(pred)) => and(&pos, &pred).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to combine positional and equality delete masks",
                )
                .with_source(e)
            })?,
        };

        filter_record_batch(&batch, &mask).map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                "Failed to apply merge-on-read deletes to a data batch",
            )
            .with_source(e)
        })
    }
}

/// Coerce a three-valued keep-mask to two-valued: every NULL becomes `false` (drop the row),
/// matching the Parquet `RowFilter` (which never keeps a null result). Mirrors the reader's
/// `coerce_nulls_to_false`. Defense in depth: `evaluate_predicate_to_mask` now returns
/// two-valued masks (Java nulls-first verdicts baked in), so this is a no-op on its output.
fn coerce_nulls_to_false(mask: &BooleanArray) -> BooleanArray {
    if mask.null_count() == 0 {
        return mask.clone();
    }
    BooleanArray::from_iter((0..mask.len()).map(|i| Some(mask.is_valid(i) && mask.value(i))))
}

pub(crate) fn is_equality_delete(f: &FileScanTaskDeleteFile) -> bool {
    matches!(f.file_type, DataContentType::EqualityDeletes)
}

/// Builds a positional-delete keep-mask for the absolute row window `[base, base + num_rows)`:
/// index `i` is `false` iff position `base + i` is a deleted position, `true` otherwise.
///
/// This is byte-identical to the naive per-row probe
/// `BooleanArray::from((0..num_rows).map(|i| !deletes.contains(base + i as u64)))`, but runs in
/// `O(D_window)` (the number of deletes falling inside the window) instead of `O(num_rows)` membership
/// probes, by range-walking the ascending [`DeleteVectorIterator`] rather than calling
/// [`DeleteVector::contains`] once per row. This is the same range-walk the Parquet path uses in
/// `ArrowReader::build_deletes_row_selection`; here it serves the Avro/ORC whole-file decode path,
/// which applies deletes post-materialization to an already-decoded batch.
///
/// ## The prime / conditional-`advance_to` / refresh dance (do not reorder)
///
/// [`DeleteVectorIterator::advance_to`] has three sharp edges this routine must respect:
///
/// 1. It is a **no-op until the iterator has been primed** with at least one `next()` — it returns
///    early while `inner` is `None`. So we call `next()` once *before* any `advance_to`.
/// 2. It repositions the *underlying* iterator but cannot un-yield a value already pulled into our
///    local `cached`. So a primed `cached` that is already a **legitimate in-window** position
///    (`>= base`) must NOT be dropped — `advance_to` cannot rewind, so discarding it would lose a
///    real delete.
/// 3. `advance_to(base)` is a **hint, not a guarantee** of landing in-window: when *no* delete
///    reaches `base`'s high-bits group (every remaining delete is below the window), it walks
///    `outer` to exhaustion and returns, leaving the iterator on a still-below-window value. So the
///    post-advance `next()` may still yield `pos < base`.
///
/// (`advance_to`'s postcondition — the next yielded position is the smallest delete `>= base` — now
/// holds across "gap groups", a high-bits group absent from the treemap between `base`'s group and a
/// present higher group; its gap-group/overshoot contract is documented on the method. An earlier
/// revision of this routine had to dodge a gap-overshoot bug in `advance_to`; that root cause is now
/// fixed, so the guard below is defense-in-depth rather than a correctness requirement.)
///
/// We call `advance_to(base)` **only when the primed `cached` is below the window** (`cached < base`)
/// — exactly the case where skipping is needed — and refresh `cached` afterward. When `cached` is
/// already `>= base` (or `None`), the iterator is already positioned and we leave it untouched.
/// Because of edge 3, the walk loop then re-checks each `pos` against `base` and only flips when
/// `base <= pos < end`, silently skipping any residual below-window delete `advance_to` could not get
/// past — a backstop that keeps the mask correct even if `advance_to` ever regresses. This is a
/// strict superset of `build_deletes_row_selection`'s stale-cache refresh, hardened to be correct
/// regardless of how far `advance_to` actually managed to skip.
///
/// `base + num_rows` is computed with `saturating_add` so a window abutting `u64::MAX` cannot wrap;
/// the `(pos - base) as usize` index is bounded by `pos < end <= base + num_rows`, so `pos - base <
/// num_rows` and the cast cannot truncate.
pub(crate) fn positional_delete_keep_mask(
    deletes: &DeleteVector,
    base: u64,
    num_rows: usize,
) -> BooleanArray {
    let mut keep = vec![true; num_rows];
    if num_rows == 0 {
        return BooleanArray::from(keep);
    }
    let end = base.saturating_add(num_rows as u64);

    let mut iter = deletes.iter();
    // PRIME: advance_to is a no-op until the iterator has yielded at least once.
    let mut cached = iter.next();
    // Best-effort fast-skip past deletes below the window — but ONLY when the primed value predates
    // the window, which keeps advance_to driven strictly forward (edge 3 above). advance_to is a
    // *hint*, not a guarantee: when no delete reaches `base`'s high-bits group it leaves the iterator
    // on a still-below-window value, so the loop below re-checks `pos < base` and never trusts
    // advance_to to land us in-window. An in-window (>= base) primed value is the first real delete
    // and is left untouched (advance_to cannot rewind).
    if let Some(pos) = cached
        && pos < base
    {
        iter.advance_to(base);
        cached = iter.next();
    }

    while let Some(pos) = cached {
        if pos >= end {
            break;
        }
        if pos >= base {
            // pos is in [base, end); pos - base < num_rows, so the index is in bounds.
            keep[(pos - base) as usize] = false;
        }
        // else pos < base: a residual below-window delete advance_to could not skip past — drop it
        // (it does not belong to this window) and keep walking; the iterator is ascending.
        cached = iter.next();
    }

    BooleanArray::from(keep)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fs::File;
    use std::ops::Not;
    use std::path::Path;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::Schema as ArrowSchema;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
    use crate::expr::{Bind, Reference};
    use crate::io::FileIO;
    use crate::spec::{DataFileFormat, Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type};

    type ArrowSchemaRef = Arc<ArrowSchema>;

    const FIELD_ID_POSITIONAL_DELETE_FILE_PATH: u64 = 2147483546;
    const FIELD_ID_POSITIONAL_DELETE_POS: u64 = 2147483545;

    #[tokio::test]
    async fn test_delete_file_filter_load_deletes() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        let delete_file_loader = CachingDeleteFileLoader::new(file_io.clone(), 10);

        let file_scan_tasks = setup(table_location);

        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_tasks[0].deletes, file_scan_tasks[0].schema_ref())
            .await
            .unwrap()
            .unwrap();

        let result = delete_filter
            .get_delete_vector(&file_scan_tasks[0])
            .unwrap();
        assert_eq!(result.len(), 12); // pos dels from pos del file 1 and 2

        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_tasks[1].deletes, file_scan_tasks[1].schema_ref())
            .await
            .unwrap()
            .unwrap();

        let result = delete_filter
            .get_delete_vector(&file_scan_tasks[1])
            .unwrap();
        assert_eq!(result.len(), 8); // no pos dels for file 3
    }

    pub(crate) fn setup(table_location: &Path) -> Vec<FileScanTask> {
        let data_file_schema = Arc::new(Schema::builder().build().unwrap());
        let positional_delete_schema = create_pos_del_schema();

        let file_path_values = [
            vec![format!("{}/1.parquet", table_location.to_str().unwrap()); 8],
            vec![format!("{}/1.parquet", table_location.to_str().unwrap()); 8],
            vec![format!("{}/2.parquet", table_location.to_str().unwrap()); 8],
        ];
        let pos_values = [
            vec![0i64, 1, 3, 5, 6, 8, 1022, 1023],
            vec![0i64, 1, 3, 5, 20, 21, 22, 23],
            vec![0i64, 1, 3, 5, 6, 8, 1022, 1023],
        ];

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        for n in 1..=3 {
            let file_path_vals = file_path_values.get(n - 1).unwrap();
            let file_path_col = Arc::new(StringArray::from_iter_values(file_path_vals));

            let pos_vals = pos_values.get(n - 1).unwrap();
            let pos_col = Arc::new(Int64Array::from_iter_values(pos_vals.clone()));

            let positional_deletes_to_write =
                RecordBatch::try_new(positional_delete_schema.clone(), vec![
                    file_path_col.clone(),
                    pos_col.clone(),
                ])
                .unwrap();

            let file = File::create(format!(
                "{}/pos-del-{}.parquet",
                table_location.to_str().unwrap(),
                n
            ))
            .unwrap();
            let mut writer = ArrowWriter::try_new(
                file,
                positional_deletes_to_write.schema(),
                Some(props.clone()),
            )
            .unwrap();

            writer
                .write(&positional_deletes_to_write)
                .expect("Writing batch");

            // writer must be closed to write footer
            writer.close().unwrap();
        }

        let pos_del_1 = FileScanTaskDeleteFile {
            file_path: format!("{}/pos-del-1.parquet", table_location.to_str().unwrap()),
            file_size_in_bytes: std::fs::metadata(format!(
                "{}/pos-del-1.parquet",
                table_location.to_str().unwrap()
            ))
            .unwrap()
            .len(),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };

        let pos_del_2 = FileScanTaskDeleteFile {
            file_path: format!("{}/pos-del-2.parquet", table_location.to_str().unwrap()),
            file_size_in_bytes: std::fs::metadata(format!(
                "{}/pos-del-2.parquet",
                table_location.to_str().unwrap()
            ))
            .unwrap()
            .len(),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };

        let pos_del_3 = FileScanTaskDeleteFile {
            file_path: format!("{}/pos-del-3.parquet", table_location.to_str().unwrap()),
            file_size_in_bytes: std::fs::metadata(format!(
                "{}/pos-del-3.parquet",
                table_location.to_str().unwrap()
            ))
            .unwrap()
            .len(),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };

        let file_scan_tasks = vec![
            FileScanTask {
                file_size_in_bytes: 0,
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!(
                    "{}/1.parquet",
                    table_location.to_str().unwrap()
                )),
                data_file_format: DataFileFormat::Parquet,
                schema: data_file_schema.clone(),
                project_field_ids: Arc::from(vec![]),
                predicate: None,
                deletes: Arc::from(vec![pos_del_1, pos_del_2.clone()]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
            },
            FileScanTask {
                file_size_in_bytes: 0,
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!(
                    "{}/2.parquet",
                    table_location.to_str().unwrap()
                )),
                data_file_format: DataFileFormat::Parquet,
                schema: data_file_schema.clone(),
                project_field_ids: Arc::from(vec![]),
                predicate: None,
                deletes: Arc::from(vec![pos_del_3]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
            },
        ];

        file_scan_tasks
    }

    pub(crate) fn create_pos_del_schema() -> ArrowSchemaRef {
        let fields = vec![
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false)
                .with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    FIELD_ID_POSITIONAL_DELETE_FILE_PATH.to_string(),
                )])),
            arrow_schema::Field::new("pos", arrow_schema::DataType::Int64, false).with_metadata(
                HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    FIELD_ID_POSITIONAL_DELETE_POS.to_string(),
                )]),
            ),
        ];
        Arc::new(arrow_schema::Schema::new(fields))
    }

    /// Risk pinned: a `FileScanTaskDeleteFile` serialized BEFORE the deletion-vector fields
    /// existed must still deserialize — the new fields default (format → Parquet, the only
    /// delete format that existed pre-DV; everything else absent). A breaking serde change here
    /// would invalidate previously serialized scan tasks.
    #[test]
    fn test_delete_file_task_without_dv_fields_deserializes_with_defaults() {
        let pre_dv_json = r#"{
            "file_path": "old-delete.parquet",
            "file_size_in_bytes": 123,
            "file_type": "PositionDeletes",
            "partition_spec_id": 0,
            "equality_ids": null
        }"#;

        let task: FileScanTaskDeleteFile =
            serde_json::from_str(pre_dv_json).expect("pre-DV serialization must deserialize");

        assert_eq!(task.file_format, DataFileFormat::Parquet);
        assert_eq!(task.referenced_data_file, None);
        assert_eq!(task.content_offset, None);
        assert_eq!(task.content_size_in_bytes, None);
        assert_eq!(task.record_count, None);
    }

    #[tokio::test]
    async fn test_build_equality_delete_predicate_case_sensitive() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "Id", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );

        // ---------- fake FileScanTask ----------
        let task = FileScanTask {
            file_size_in_bytes: 0,
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from("data.parquet"),
            data_file_format: crate::spec::DataFileFormat::Parquet,
            schema: schema.clone(),
            project_field_ids: Arc::from(vec![]),
            predicate: None,
            deletes: Arc::from(vec![FileScanTaskDeleteFile {
                file_path: "eq-del.parquet".to_string(),
                file_size_in_bytes: 1, // never read; this test fails before opening the file
                file_type: DataContentType::EqualityDeletes,
                partition_spec_id: 0,
                equality_ids: None,
                file_format: DataFileFormat::Parquet,
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
                record_count: None,
            }]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
            split_offsets: None,
        };

        let filter = DeleteFilter::default();

        // ---------- insert equality delete predicate ----------
        let pred = Reference::new("id").equal_to(Datum::long(10));

        let (tx, rx) = tokio::sync::oneshot::channel();
        filter
            .try_start_eq_del_load("eq-del.parquet")
            .expect("a fresh eq-delete file must be claimable")
            .spawn_publisher(rx);

        // No key set (predicate-only path) for this case-sensitivity test.
        tx.send((pred, None)).unwrap();

        // ---------- should FAIL ----------
        // BOUNDED: this call reaches the eq-delete wait path, so a lost-wakeup regression makes it
        // never return. Without the timeout that is a hung CI job instead of a red test (the
        // eq-delete arming mutations do exactly that) — the same bound
        // `test_failed_eq_delete_load_surfaces_error_not_hang` already carries.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            filter.build_equality_delete_predicate(&task),
        )
        .await
        .expect("build_equality_delete_predicate must not hang");

        assert!(
            result.is_err(),
            "case_sensitive=true should fail when column case mismatches"
        );
    }

    /// The public engine-facing surface: `DeleteFilter::load` (hiding the loader) -> the position
    /// accessor -> `apply` on a batch the engine read itself. Same fixture as
    /// `test_delete_file_filter_load_deletes`.
    #[tokio::test]
    async fn test_public_delete_filter_load_and_apply() {
        use arrow_array::Array;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();
        let tasks = setup(table_location);

        // Public constructor — resolves the task's deletes without touching CachingDeleteFileLoader.
        let filter = DeleteFilter::load(&tasks[0], file_io).await.unwrap();

        // Positional deletes for data file 1: {0,1,3,5,6,8,20,21,22,23,1022,1023} = 12 distinct.
        let positions = filter.deleted_row_positions(&tasks[0]).unwrap();
        assert_eq!(positions.len(), 12);
        // The fixture has no equality deletes.
        assert!(
            filter
                .equality_delete_predicate(&tasks[0])
                .await
                .unwrap()
                .is_none()
        );

        // Apply to a 10-row batch (file positions 0..9). Deleted in that window: {0,1,3,5,6,8} =>
        // survivors {2,4,7,9}.
        let field =
            arrow_schema::Field::new("x", arrow_schema::DataType::Int64, false).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]),
            );
        let schema = Arc::new(ArrowSchema::new(vec![field]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from_iter_values(
            0i64..10,
        ))])
        .unwrap();

        let surviving = filter.apply(&tasks[0], batch, 0, None).unwrap();
        let col = surviving
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(col.values(), &[2, 4, 7, 9]);
    }

    /// Risk pinned (audit SAF-001 / BUG-004): a FAILED equality-delete load must surface a typed
    /// error to the waiting consumer within a BOUNDED await — it must never hang. When the loader
    /// drops the oneshot sender without sending (the shape of every early-return in the
    /// load → parse → send window), the receiver task transitions the entry to `EqDelState::Failed`
    /// and STILL wakes the waiters; the waiter reads `Failed` as absence and the predicate builder
    /// raises a typed error. MUTATION: reverting the terminal transition (back to
    /// `eq_del.await.unwrap()`) leaves the entry `Loading` forever and this test TIMES OUT (RED).
    /// Deterministic on the default current-thread test runtime: the consumer registers its
    /// notifier interest (first poll of `notified()`) before the dropped-sender receiver task is
    /// scheduled, so the terminal `notify_waiters()` cannot be missed.
    #[tokio::test]
    async fn test_failed_eq_delete_load_surfaces_error_not_hang() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .expect("schema"),
        );

        let task = FileScanTask {
            file_size_in_bytes: 0,
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from("data.parquet"),
            data_file_format: DataFileFormat::Parquet,
            schema: schema.clone(),
            project_field_ids: Arc::from(vec![]),
            predicate: None,
            deletes: Arc::from(vec![FileScanTaskDeleteFile {
                file_path: "eq-del.parquet".to_string(),
                file_size_in_bytes: 1,
                file_type: DataContentType::EqualityDeletes,
                partition_spec_id: 0,
                equality_ids: Some(vec![1]),
                file_format: DataFileFormat::Parquet,
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
                record_count: None,
            }]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
        };

        let filter = DeleteFilter::default();
        let (tx, rx) = tokio::sync::oneshot::channel::<(Predicate, Option<EqDeleteKeySet>)>();
        filter
            .try_start_eq_del_load("eq-del.parquet")
            .expect("a fresh eq-delete file must be claimable")
            .spawn_publisher(rx);

        // Simulate the loader failing AFTER registration: the parsed predicate is never sent and
        // the sender is dropped — exactly what an early-return in the load window does.
        drop(tx);

        // The consumer must get a typed error INSIDE the timeout, never block on the notifier.
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            filter.build_equality_delete_predicate(&task),
        )
        .await;

        let built =
            outcome.expect("build_equality_delete_predicate must not hang after a load failure");
        let error =
            built.expect_err("a failed eq-delete load must surface a typed error to the consumer");
        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert!(
            error.to_string().contains("eq-del.parquet"),
            "error must name the eq-delete file: {error}"
        );
    }

    /// Build a parquet positional-delete task entry for `file_path` (metadata only — these tests
    /// never open it).
    fn parquet_pos_del_entry(file_path: &str) -> FileScanTaskDeleteFile {
        FileScanTaskDeleteFile {
            file_path: file_path.to_string(),
            file_size_in_bytes: 0,
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        }
    }

    /// Risk pinned (audit SAF-003): a thread that panics while holding the `state` write guard
    /// poisons the `RwLock`, but subsequent scan operations must RECOVER (`into_inner`) rather than
    /// cascade-panic. MUTATION: restoring `self.state.write().unwrap()` in
    /// `install_pos_del_contribution` / `try_start_pos_del_load` turns these calls into panics on
    /// the poisoned lock (RED). (Adapted for R117: the writer under test was `upsert_delete_vector`
    /// until the per-task scoping change replaced it with `install_pos_del_contribution` — the
    /// pinned risk, poison-recovery on the state writers, is unchanged.)
    #[test]
    fn test_poisoned_state_lock_recovers_instead_of_cascading() {
        let filter = DeleteFilter::default();
        // Claim BEFORE poisoning so only the writer-under-test runs on the poisoned lock.
        let guard = claim_pos_del(&filter, "pos-del-installed.parquet");

        // Poison the shared state RwLock by panicking while holding the write guard.
        let poisoner = filter.clone();
        let handle = std::thread::spawn(move || {
            let _guard = poisoner
                .state
                .write()
                .expect("acquire write guard to poison");
            panic!("intentionally poison the delete-filter state lock");
        });
        assert!(
            handle.join().is_err(),
            "the poisoning thread must have panicked while holding the guard"
        );

        // A subsequent WRITER (install_pos_del_contribution) must not panic on the poisoned lock,
        // and its write must land.
        filter.install_pos_del_contribution(
            &guard,
            HashMap::from([("data.parquet".to_string(), DeleteVector::default())]),
        );
        assert!(
            recover_poison(filter.state.read())
                .pos_del_contributions
                .contains_key("pos-del-installed.parquet"),
            "the recovered write must land despite the poisoned lock"
        );
        // Publishing the claim also runs on the poisoned lock (the guard's own recover path).
        guard.publish_loaded();

        // A subsequent writer via try_start_pos_del_load must also recover and proceed.
        assert!(
            matches!(
                filter
                    .try_start_pos_del_load("pos-del.parquet")
                    .expect("claiming a fresh file must not error"),
                PosDelLoadAction::Load(_)
            ),
            "a fresh positional-delete load must proceed on the recovered lock"
        );
    }

    /// Risk pinned (G1a fail-open): a poisoned `state` lock must NOT make
    /// `resolve_delete_vector` return `None` for a present contribution. A `None` is read by the
    /// reader / `apply` as "no positional deletes here", so a poison-induced `None` would silently
    /// DROP the file's positional deletes and RESURRECT deleted rows. The resolver must recover the
    /// poison (`into_inner`) and still hand back the frozen delete vector.
    /// MUTATION: reverting the resolver's state read to `self.state.read().ok()` + early-`None`
    /// swallows the poison as `None` and this test FAILS (the `expect` below trips) — RED.
    /// (Adapted for R117: the accessor under test was `get_delete_vector_for_path` until the
    /// per-task scoping change replaced it with the task-scoped resolver — the pinned risk, a
    /// poisoned read failing open as "no deletes", is unchanged. FK3: memoized vectors are
    /// `Arc<DeleteVector>` — poison recovery remains only on the outer state `RwLock`.)
    #[test]
    fn test_get_delete_vector_survives_poisoned_lock() {
        let filter = DeleteFilter::default();

        // Populate a contribution for a data file so a correct read returns `Some`, installing it
        // exactly as the production loader does (claim → install → publish).
        let mut dv = DeleteVector::default();
        dv.insert(7);
        let guard = claim_pos_del(&filter, "pos-del.parquet");
        filter.install_pos_del_contribution(
            &guard,
            HashMap::from([("data.parquet".to_string(), dv)]),
        );
        guard.publish_loaded();

        // Poison the shared state RwLock by panicking while holding the write guard.
        let poisoner = filter.clone();
        let handle = std::thread::spawn(move || {
            let _guard = poisoner
                .state
                .write()
                .expect("acquire write guard to poison");
            panic!("intentionally poison the delete-filter state lock");
        });
        assert!(
            handle.join().is_err(),
            "the poisoning thread must have panicked while holding the guard"
        );

        // The resolver must RECOVER the poison and still return the present delete vector — not
        // swallow the poison as `None` (which would resurrect deleted row 7).
        let dv = filter
            .resolve_delete_vector(&[parquet_pos_del_entry("pos-del.parquet")], "data.parquet")
            .expect("a present delete vector must survive a poisoned state lock, not read as None");
        assert!(
            dv.contains(7),
            "the recovered delete vector must still carry its deleted positions"
        );
    }

    /// FK3 / scout #12: memoized positional vectors freeze as `Arc<DeleteVector>` and are shared
    /// by pointer across resolvers of the same task shape. MUTATION: re-wrap every resolve in a
    /// fresh `Arc::new(...)` (no memo install, or clone the inner bitmap into a new Arc each
    /// time) turns `Arc::ptr_eq` RED.
    #[test]
    fn test_resolved_pos_del_vector_is_frozen_arc_shared() {
        let filter = DeleteFilter::default();
        let mut dv = DeleteVector::default();
        dv.insert(1);
        dv.insert(3);
        let guard = claim_pos_del(&filter, "pos-del.parquet");
        filter.install_pos_del_contribution(
            &guard,
            HashMap::from([("data.parquet".to_string(), dv)]),
        );
        guard.publish_loaded();

        let deletes = [parquet_pos_del_entry("pos-del.parquet")];
        let a = filter
            .resolve_delete_vector(&deletes, "data.parquet")
            .expect("vector must resolve");
        let b = filter
            .resolve_delete_vector(&deletes, "data.parquet")
            .expect("vector must resolve again");
        assert!(
            Arc::ptr_eq(&a, &b),
            "repeated resolve of one task shape must share one frozen Arc"
        );
        assert_eq!(a.len(), 2);
        assert!(a.contains(1) && a.contains(3));
    }

    /// FK3 / scout #12: multi-source resolve ORs by reference (no per-contribution roaring clone)
    /// into one frozen Arc. Positions from BOTH sources must appear; a second resolve of the
    /// same key shares the Arc. MUTATION: only merging the first contribution (skip the loop's
    /// later sources) turns the cardinality/contains asserts RED.
    #[test]
    fn test_multi_source_resolve_ors_by_ref_into_frozen_arc() {
        let filter = DeleteFilter::default();

        let mut dv_a = DeleteVector::default();
        dv_a.insert(1);
        dv_a.insert(5);
        let guard_a = claim_pos_del(&filter, "pos-a.parquet");
        filter.install_pos_del_contribution(
            &guard_a,
            HashMap::from([("data.parquet".to_string(), dv_a)]),
        );
        guard_a.publish_loaded();

        let mut dv_b = DeleteVector::default();
        dv_b.insert(5); // overlap with A
        dv_b.insert(9);
        let guard_b = claim_pos_del(&filter, "pos-b.parquet");
        filter.install_pos_del_contribution(
            &guard_b,
            HashMap::from([("data.parquet".to_string(), dv_b)]),
        );
        guard_b.publish_loaded();

        let deletes = [
            parquet_pos_del_entry("pos-a.parquet"),
            parquet_pos_del_entry("pos-b.parquet"),
        ];
        let merged = filter
            .resolve_delete_vector(&deletes, "data.parquet")
            .expect("union must resolve");
        assert_eq!(merged.len(), 3, "union of {{1,5}} and {{5,9}} is {{1,5,9}}");
        assert!(merged.contains(1) && merged.contains(5) && merged.contains(9));

        let again = filter
            .resolve_delete_vector(&deletes, "data.parquet")
            .expect("memoized re-resolve");
        assert!(
            Arc::ptr_eq(&merged, &again),
            "multi-source memo must freeze as one shared Arc"
        );
    }

    /// EQ-DELETE SWEEP for the R117 class (documents that the equality path does NOT share the
    /// cross-task defect): equality-delete state is a pure load-cache keyed by DELETE FILE path,
    /// and APPLICATION (`build_equality_delete_predicate`, `collect_equality_delete_keysets`)
    /// iterates `task.deletes` — so a predicate loaded for one task can never fold into a task
    /// that does not list its file. Mirrors Java: `DeleteFilter.applyEqDeletes()` reads only
    /// `this.eqDeletes`, the constructor's partition of the task's own list (1.10.0 bytecode,
    /// constructor offsets 51-208).
    ///
    /// Two eq-delete files are loaded into ONE shared filter; the task listing only the first must
    /// get exactly the first's predicate (never the second's), and a task listing neither gets
    /// `None`. MUTATION: folding every loaded eq predicate from the shared state into the task's
    /// predicate (ignoring `task.deletes`) turns this RED on the `id = 20` assertion.
    #[tokio::test]
    async fn test_eq_delete_application_scoped_to_tasks_own_files() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .expect("schema"),
        );

        let eq_delete_entry = |file_path: &str| FileScanTaskDeleteFile {
            file_path: file_path.to_string(),
            file_size_in_bytes: 1,
            file_type: DataContentType::EqualityDeletes,
            partition_spec_id: 0,
            equality_ids: Some(vec![1]),
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };
        let task_for = |data_file_path: &str, deletes: Vec<FileScanTaskDeleteFile>| FileScanTask {
            file_size_in_bytes: 0,
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from(data_file_path),
            data_file_format: DataFileFormat::Parquet,
            schema: schema.clone(),
            project_field_ids: Arc::from(vec![]),
            predicate: None,
            deletes: Arc::from(deletes),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
        };

        let filter = DeleteFilter::default();
        // Load TWO eq-delete predicates into the ONE shared filter, through the production claim +
        // publish machinery.
        for (path, value) in [("eq-del-1.parquet", 10i64), ("eq-del-2.parquet", 20i64)] {
            let (tx, rx) = tokio::sync::oneshot::channel();
            filter
                .try_start_eq_del_load(path)
                .expect("a fresh eq-delete file must be claimable")
                .spawn_publisher(rx);
            tx.send((Reference::new("id").equal_to(Datum::long(value)), None))
                .expect("the publisher task must be listening");
        }

        // The task lists ONLY eq-del-1: its combined predicate must be exactly eq-del-1's.
        let task_with_first = task_for("data-x.parquet", vec![eq_delete_entry("eq-del-1.parquet")]);
        let bound = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            filter.build_equality_delete_predicate(&task_with_first),
        )
        .await
        .expect("predicate build must not hang")
        .expect("predicate build must succeed")
        .expect("a task with an eq delete must get a predicate");
        let rendered = bound.to_string();
        assert!(
            rendered.contains("10"),
            "the task's own eq-delete predicate must be applied, got: {rendered}"
        );
        assert!(
            !rendered.contains("20"),
            "eq-del-2 (loaded for ANOTHER task, absent from this task's delete list) must not \
             fold into this task's predicate, got: {rendered}"
        );

        // A task listing NO eq deletes gets no predicate at all, however much the shared state
        // holds.
        let task_without = task_for("data-y.parquet", vec![]);
        let none = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            filter.build_equality_delete_predicate(&task_without),
        )
        .await
        .expect("predicate build must not hang")
        .expect("predicate build must succeed");
        assert!(
            none.is_none(),
            "a task with no eq deletes must get None even when the shared state holds predicates"
        );
    }

    // =============================================================================================
    // H6 equivalence harness — eq-delete SET membership vs the production PREDICATE path.
    //
    // The production equality-delete application builds, per delete row, a leaf predicate
    // (`col = v` for a non-null cell, `col IS NULL` for a null cell), AND-folds the cells, negates
    // per row, AND-folds the rows, binds, evaluates the bound predicate over the data batch with the
    // arrow comparison kernels, and coerces NULL results to `false`. A data row is DELETED iff that
    // evaluation makes the survival predicate FALSE — i.e. iff the row matches some delete tuple
    // under ARROW `eq` semantics.
    //
    // These tests pin the EXACT semantics any O(R) set-membership rewrite (the H6 optimization) would
    // have to reproduce byte-for-byte, and demonstrate WHERE a naive `HashSet<Datum>` set diverges
    // from that oracle — the evidence behind deferring H6 (see the build summary).
    // =============================================================================================

    /// The production "deleted" mask oracle for a single-column eq-delete: build the survival
    /// predicate exactly as `parse_equality_deletes_record_batch_stream` does, bind it, evaluate it
    /// over `data_batch`, coerce nulls to false, and return `deleted[i] = !survives[i]`.
    fn oracle_deleted_mask(
        col_name: &str,
        schema: SchemaRef,
        delete_cells: &[Option<Datum>],
        data_batch: &RecordBatch,
    ) -> Vec<bool> {
        // Per-delete-row survival predicate: NOT(col = v) / NOT(col IS NULL), exactly as production.
        let mut row_predicates: Vec<Predicate> = Vec::new();
        for cell in delete_cells {
            let leaf = match cell {
                Some(datum) => Reference::new(col_name).equal_to(datum.clone()),
                None => Reference::new(col_name).is_null(),
            };
            row_predicates.push(leaf.not().rewrite_not());
        }
        // Balanced AND-fold of the survival predicates (matches production's tree builder).
        while row_predicates.len() > 1 {
            let mut next = Vec::with_capacity(row_predicates.len().div_ceil(2));
            let mut it = row_predicates.into_iter();
            while let Some(p1) = it.next() {
                match it.next() {
                    Some(p2) => next.push(p1.and(p2)),
                    None => next.push(p1),
                }
            }
            row_predicates = next;
        }
        let survival = row_predicates.pop().unwrap_or(AlwaysTrue);
        let bound = survival
            .bind(schema, false)
            .expect("bind survival predicate");
        let survives = coerce_nulls_to_false(
            &evaluate_predicate_to_mask(&bound, data_batch).expect("evaluate survival mask"),
        );
        (0..survives.len()).map(|i| !survives.value(i)).collect()
    }

    /// Candidate O(R) set path for a SINGLE column: insert each non-null delete value into a
    /// `HashSet<Datum>` (and remember whether any delete cell is null); a data row is deleted iff its
    /// value is in the set, or it is null and a null delete cell exists. This is the obvious
    /// set-membership rewrite — the tests below show exactly when it agrees with the oracle and when
    /// it does NOT.
    fn candidate_set_deleted_mask(
        delete_cells: &[Option<Datum>],
        data_cells: &[Option<Datum>],
    ) -> Vec<bool> {
        let mut set: std::collections::HashSet<Datum> = std::collections::HashSet::new();
        let mut has_null_delete = false;
        for cell in delete_cells {
            match cell {
                Some(d) => {
                    set.insert(d.clone());
                }
                None => has_null_delete = true,
            }
        }
        data_cells
            .iter()
            .map(|cell| match cell {
                Some(d) => set.contains(d),
                None => has_null_delete,
            })
            .collect()
    }

    fn long_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "v", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn double_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "v", Type::Primitive(PrimitiveType::Double)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn long_batch(values: &[Option<i64>]) -> RecordBatch {
        let field =
            arrow_schema::Field::new("v", arrow_schema::DataType::Int64, true).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]),
            );
        let schema = Arc::new(ArrowSchema::new(vec![field]));
        let col = Int64Array::from(values.to_vec());
        RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
    }

    fn double_batch(values: &[Option<f64>]) -> RecordBatch {
        use arrow_array::Float64Array;
        let field =
            arrow_schema::Field::new("v", arrow_schema::DataType::Float64, true).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]),
            );
        let schema = Arc::new(ArrowSchema::new(vec![field]));
        let col = Float64Array::from(values.to_vec());
        RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
    }

    /// PROVABLE-SAFE case: integers, including NULL delete and NULL data rows, duplicate delete keys,
    /// all-match, none-match. The set path agrees with the oracle EXACTLY here — integers have no
    /// NaN/-0.0 hazard, and the production path's `col IS NULL` leaf coincides with set null
    /// handling. (This is the slice of inputs an H6 fast path COULD safely cover.)
    #[test]
    fn test_h6_equivalence_long_with_nulls_and_dups_matches() {
        let schema = long_schema();
        // delete tuples: 3, 3 (dup), 7, NULL
        let delete_cells = vec![
            Some(Datum::long(3)),
            Some(Datum::long(3)),
            Some(Datum::long(7)),
            None,
        ];
        // data rows: 3, 7, 9, NULL, 100
        let data_vals = vec![Some(3i64), Some(7), Some(9), None, Some(100)];
        let data_cells: Vec<Option<Datum>> = data_vals.iter().map(|v| v.map(Datum::long)).collect();
        let batch = long_batch(&data_vals);

        let oracle = oracle_deleted_mask("v", schema, &delete_cells, &batch);
        let candidate = candidate_set_deleted_mask(&delete_cells, &data_cells);

        assert_eq!(
            oracle, candidate,
            "integer eq-delete: set path must match the predicate oracle exactly"
        );
        // Pin the expected mask too (3 deleted, 7 deleted, 9 kept, NULL deleted by NULL delete, 100 kept).
        assert_eq!(oracle, vec![true, true, false, true, false]);
    }

    /// DIVERGENCE PROOF — `-0.0` / `+0.0` (the H6 deferral evidence): the production path compares
    /// floats via `arrow_ord::cmp::eq`, whose float kernels use TOTAL ordering — `-0.0` and `+0.0`
    /// are DISTINCT, so a `+0.0` delete deletes `+0.0` but NOT `-0.0`. A `HashSet<Datum>` keyed on
    /// `OrderedFloat` instead COLLAPSES `-0.0` and `+0.0` into one key (they hash and compare equal),
    /// so the naive set path would ALSO delete the `-0.0` row. The masks differ on row 0. This is the
    /// concrete reason a naive `HashSet<Datum>` set rewrite is UNSOUND vs the current predicate path:
    /// it would change which rows are deleted on signed-zero float keys.
    #[test]
    fn test_h6_naive_set_diverges_on_negative_zero() {
        let schema = double_schema();
        let delete_cells = vec![Some(Datum::double(0.0f64))]; // delete value +0.0
        let data_vals = vec![Some(-0.0f64), Some(0.0f64), Some(1.0f64)];
        let data_cells: Vec<Option<Datum>> =
            data_vals.iter().map(|v| v.map(Datum::double)).collect();
        let batch = double_batch(&data_vals);

        let oracle = oracle_deleted_mask("v", schema, &delete_cells, &batch);
        let candidate = candidate_set_deleted_mask(&delete_cells, &data_cells);

        // Oracle (arrow total-ordering eq): only +0.0 is deleted; -0.0 is a distinct value (kept).
        assert_eq!(
            oracle,
            vec![false, true, false],
            "total-ordering eq distinguishes -0.0 from +0.0: only +0.0 deleted"
        );
        // Naive set (OrderedFloat collapses ±0.0): -0.0 AND +0.0 both deleted — the divergence.
        assert_eq!(candidate, vec![true, true, false]);
        assert_ne!(
            oracle, candidate,
            "the naive HashSet<Datum> set path MUST diverge from the predicate oracle on signed \
             zero; this proves H6 cannot ship a naive set without matching arrow's total-ordering \
             float equality exactly"
        );
    }

    /// EQUIVALENCE — `NaN`: `arrow_ord::cmp::eq`'s total-ordering float kernel treats `NaN == NaN` as
    /// TRUE (every NaN bit-pattern collapses to the canonical NaN under total ordering), so a `NaN`
    /// delete DOES delete a `NaN` data row. A `HashSet<Datum>` keyed on `OrderedFloat` also treats
    /// `NaN == NaN`, so the paths agree. (Both differ from Java `StructLikeSet`, which is bit-wise —
    /// but the prompt's oracle is the CURRENT Rust path, which these tests pin.)
    #[test]
    fn test_h6_set_matches_predicate_on_nan() {
        let schema = double_schema();
        let delete_cells = vec![Some(Datum::double(f64::NAN))];
        let data_vals = vec![Some(f64::NAN), Some(1.0f64)];
        let data_cells: Vec<Option<Datum>> =
            data_vals.iter().map(|v| v.map(Datum::double)).collect();
        let batch = double_batch(&data_vals);

        let oracle = oracle_deleted_mask("v", schema, &delete_cells, &batch);
        let candidate = candidate_set_deleted_mask(&delete_cells, &data_cells);

        // Both paths: the NaN row IS deleted by a NaN delete (total ordering: NaN == NaN).
        assert_eq!(
            oracle,
            vec![true, false],
            "total-ordering eq matches NaN == NaN, so a NaN delete deletes the NaN row"
        );
        assert_eq!(
            oracle, candidate,
            "the HashSet<Datum> set path matches the predicate oracle on NaN"
        );
    }

    // =============================================================================================
    // SOUND H6 — the REAL `EqDeleteKeySet` fast path proven byte-identical to the predicate ORACLE
    // across the full NON-FLOAT type matrix (single- and multi-column), and the type GATE proven to
    // route Float/Double back to the (untouched) predicate path.
    //
    // Each test builds a data batch + schema, a set of delete tuples, runs BOTH:
    //   * the predicate oracle (`multi_col_oracle_deleted_mask`) — production's per-delete-row
    //     survival predicate, bound, evaluated, nulls-coerced, negated → the deleted mask, and
    //   * the production `EqDeleteKeySet::delete_mask` (the fast path),
    // and asserts the masks are IDENTICAL. The delete tuples and the predicate leaves are produced
    // from the SAME `Datum`s, and `delete_mask` decodes the data column with the SAME
    // `arrow_primitive_to_literal` conversion the predicate path's columns use — so the only thing
    // under test is that `Datum` equality matches the Arrow `eq` kernel for the admitted types.
    // =============================================================================================

    /// Multi-column predicate oracle: a row is DELETED iff it matches some delete tuple under the
    /// production survival predicate `AND over files NOT(AND over cols col_i = v_i / col_i IS NULL)`.
    /// Builds exactly the predicate `parse_equality_deletes_record_batch_stream` builds for one file.
    fn multi_col_oracle_deleted_mask(
        col_names: &[&str],
        schema: SchemaRef,
        delete_rows: &[Vec<Option<Datum>>],
        data_batch: &RecordBatch,
    ) -> Vec<bool> {
        let mut row_predicates: Vec<Predicate> = Vec::new();
        for row in delete_rows {
            let mut row_pred = AlwaysTrue;
            for (cell, name) in row.iter().zip(col_names.iter()) {
                let leaf = match cell {
                    Some(datum) => Reference::new(*name).equal_to(datum.clone()),
                    None => Reference::new(*name).is_null(),
                };
                row_pred = row_pred.and(leaf);
            }
            row_predicates.push(row_pred.not().rewrite_not());
        }
        while row_predicates.len() > 1 {
            let mut next = Vec::with_capacity(row_predicates.len().div_ceil(2));
            let mut it = row_predicates.into_iter();
            while let Some(p1) = it.next() {
                match it.next() {
                    Some(p2) => next.push(p1.and(p2)),
                    None => next.push(p1),
                }
            }
            row_predicates = next;
        }
        let survival = row_predicates.pop().unwrap_or(AlwaysTrue);
        let bound = survival
            .bind(schema, false)
            .expect("bind survival predicate");
        let survives = coerce_nulls_to_false(
            &evaluate_predicate_to_mask(&bound, data_batch).expect("evaluate survival mask"),
        );
        (0..survives.len()).map(|i| !survives.value(i)).collect()
    }

    /// Build a `RecordBatch` whose columns carry the `PARQUET_FIELD_ID_META_KEY` metadata
    /// (`field_id = position + 1`) so both the predicate evaluator and `EqDeleteKeySet::delete_mask`
    /// resolve the same columns.
    fn batch_with_field_ids(fields: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let arrow_fields: Vec<arrow_schema::Field> = fields
            .iter()
            .enumerate()
            .map(|(i, (name, arr))| {
                arrow_schema::Field::new(*name, arr.data_type().clone(), true).with_metadata(
                    HashMap::from([(
                        PARQUET_FIELD_ID_META_KEY.to_string(),
                        (i as i32 + 1).to_string(),
                    )]),
                )
            })
            .collect();
        let schema = Arc::new(ArrowSchema::new(arrow_fields));
        let columns: Vec<ArrayRef> = fields.into_iter().map(|(_, arr)| arr).collect();
        RecordBatch::try_new(schema, columns).expect("build data batch")
    }

    /// Drive the equivalence for a batch with NO NULL in any key column: assert
    /// `EqDeleteKeySet::delete_mask` returns `Some(mask)` byte-identical to the predicate oracle, and
    /// return the agreed mask so the caller can also pin its exact value.
    fn assert_set_matches_oracle(
        iceberg_schema: SchemaRef,
        key_columns: Vec<(i32, String, PrimitiveType)>,
        col_names: &[&str],
        delete_rows: Vec<Vec<Option<Datum>>>,
        data_fields: Vec<(&str, ArrayRef)>,
    ) -> Vec<bool> {
        let batch = batch_with_field_ids(data_fields);
        let oracle = multi_col_oracle_deleted_mask(col_names, iceberg_schema, &delete_rows, &batch);

        let set = EqDeleteKeySet::try_build(key_columns, delete_rows)
            .expect("non-float key columns must build a set");
        let set_mask = set
            .delete_mask(&batch)
            .expect("set delete_mask")
            .expect("a batch with no key-column NULL must take the set fast path");

        assert_eq!(
            set_mask, oracle,
            "EqDeleteKeySet fast path must equal the predicate oracle, byte-for-byte"
        );
        oracle
    }

    fn opt_schema(fields: Vec<(i32, &str, PrimitiveType)>) -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(
                    fields
                        .into_iter()
                        .map(|(id, name, ty)| {
                            NestedField::optional(id, name, Type::Primitive(ty)).into()
                        })
                        .collect::<Vec<_>>(),
                )
                .build()
                .unwrap(),
        )
    }

    /// Long key — duplicates, all-match, none-match, NULL DELETE tuple (which deletes nothing among
    /// non-null data rows). Data has no key-column NULL → the set fast path is taken.
    #[test]
    fn test_h6_set_long_matches_oracle() {
        let schema = opt_schema(vec![(1, "v", PrimitiveType::Long)]);
        let key_columns = vec![(1, "v".to_string(), PrimitiveType::Long)];
        let delete_rows = vec![
            vec![Some(Datum::long(3))],
            vec![Some(Datum::long(3))], // duplicate
            vec![Some(Datum::long(7))],
            vec![None], // NULL delete tuple
        ];
        let data: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(3i64),
            Some(7),
            Some(9),
            Some(100),
        ]));
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["v"], delete_rows, vec![("v", data)]);
        assert_eq!(mask, vec![true, true, false, false]);
    }

    /// Int key (Int32Array → I64 store) — pins i32→i64 widen on both build (`literal_as_i64`) and
    /// probe (`i64_column_values`) so the specialized store stays oracle-identical for Int.
    #[test]
    fn test_h6_set_int_matches_oracle() {
        use arrow_array::Int32Array;
        let schema = opt_schema(vec![(1, "v", PrimitiveType::Int)]);
        let key_columns = vec![(1, "v".to_string(), PrimitiveType::Int)];
        let delete_rows = vec![vec![Some(Datum::int(3))], vec![Some(Datum::int(7))]];
        let data: ArrayRef = Arc::new(Int32Array::from(vec![Some(3i32), Some(7), Some(9)]));
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["v"], delete_rows, vec![("v", data)]);
        assert_eq!(mask, vec![true, true, false]);
    }

    /// Timestamp (micros) key — I64 store path for TimestampMicrosecondArray.
    #[test]
    fn test_h6_set_timestamp_matches_oracle() {
        use arrow_array::TimestampMicrosecondArray;
        let schema = opt_schema(vec![(1, "ts", PrimitiveType::Timestamp)]);
        let key_columns = vec![(1, "ts".to_string(), PrimitiveType::Timestamp)];
        let delete_rows = vec![vec![Some(Datum::timestamp_micros(1_000_000))]];
        let data: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(1_000_000i64),
            Some(2_000_000),
        ]));
        let mask = assert_set_matches_oracle(schema, key_columns, &["ts"], delete_rows, vec![(
            "ts", data,
        )]);
        assert_eq!(mask, vec![true, false]);
    }

    /// Timestamptz (micros) — same physical I64 path as Timestamp; pins the type arm.
    #[test]
    fn test_h6_set_timestamptz_matches_oracle() {
        use arrow_array::TimestampMicrosecondArray;
        let schema = opt_schema(vec![(1, "tsz", PrimitiveType::Timestamptz)]);
        let key_columns = vec![(1, "tsz".to_string(), PrimitiveType::Timestamptz)];
        let delete_rows = vec![vec![Some(Datum::timestamptz_micros(5_000_000))]];
        let data: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(5_000_000i64), Some(6_000_000)])
                .with_timezone("UTC"),
        );
        let mask = assert_set_matches_oracle(schema, key_columns, &["tsz"], delete_rows, vec![(
            "tsz", data,
        )]);
        assert_eq!(mask, vec![true, false]);
    }

    /// TimestampNs — I64 store via TimestampNanosecondArray.
    #[test]
    fn test_h6_set_timestamp_ns_matches_oracle() {
        use arrow_array::TimestampNanosecondArray;
        let schema = opt_schema(vec![(1, "tsn", PrimitiveType::TimestampNs)]);
        let key_columns = vec![(1, "tsn".to_string(), PrimitiveType::TimestampNs)];
        let delete_rows = vec![vec![Some(Datum::timestamp_nanos(1_000_000_000))]];
        let data: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![
            Some(1_000_000_000i64),
            Some(2_000_000_000),
        ]));
        let mask = assert_set_matches_oracle(schema, key_columns, &["tsn"], delete_rows, vec![(
            "tsn", data,
        )]);
        assert_eq!(mask, vec![true, false]);
    }

    /// Multi-column Bytes store retains null tags (unlike I64). Null-only delete tuples must
    /// keep the set non-empty and null-bail so the predicate `IS NULL` leaves still apply.
    #[test]
    fn test_h6_set_multi_column_null_only_bails_on_null_data() {
        use arrow_array::StringArray;
        let schema = opt_schema(vec![
            (1, "id", PrimitiveType::Long),
            (2, "name", PrimitiveType::String),
        ]);
        let key_columns = vec![
            (1, "id".to_string(), PrimitiveType::Long),
            (2, "name".to_string(), PrimitiveType::String),
        ];
        // Full-null delete tuple — Bytes encodes TAG_NULL per cell.
        let delete_rows = vec![vec![None, None]];
        let set = EqDeleteKeySet::try_build(key_columns, delete_rows.clone()).expect("set builds");
        assert!(
            !set.is_empty(),
            "Bytes null-only multi-col set must be non-empty (null tags retained)"
        );

        let id: ArrayRef = Arc::new(Int64Array::from(vec![Some(1i64), None]));
        let name: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None]));
        let batch = batch_with_field_ids(vec![("id", id), ("name", name)]);
        assert_eq!(
            set.delete_mask(&batch).expect("delete_mask"),
            None,
            "null data must bail for Bytes null-only multi-col deletes"
        );
        let oracle = multi_col_oracle_deleted_mask(&["id", "name"], schema, &delete_rows, &batch);
        // Only the all-null data row matches the all-null delete tuple.
        assert_eq!(oracle, vec![false, true]);
    }

    /// String key — empty string, no-match. (Non-null data → set path.)
    #[test]
    fn test_h6_set_string_matches_oracle() {
        use arrow_array::StringArray;
        let schema = opt_schema(vec![(1, "s", PrimitiveType::String)]);
        let key_columns = vec![(1, "s".to_string(), PrimitiveType::String)];
        let delete_rows = vec![vec![Some(Datum::string("a"))], vec![Some(Datum::string(
            "",
        ))]];
        let data: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), Some(""), Some("z")]));
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["s"], delete_rows, vec![("s", data)]);
        assert_eq!(mask, vec![true, true, false]);
    }

    /// Boolean key.
    #[test]
    fn test_h6_set_bool_matches_oracle() {
        use arrow_array::BooleanArray as ArrowBool;
        let schema = opt_schema(vec![(1, "b", PrimitiveType::Boolean)]);
        let key_columns = vec![(1, "b".to_string(), PrimitiveType::Boolean)];
        let delete_rows = vec![vec![Some(Datum::bool(true))]];
        let data: ArrayRef = Arc::new(ArrowBool::from(vec![Some(true), Some(false)]));
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["b"], delete_rows, vec![("b", data)]);
        assert_eq!(mask, vec![true, false]);
    }

    /// Date key (Int32-backed temporal) — confirms temporal types compare as their integer backing.
    #[test]
    fn test_h6_set_date_matches_oracle() {
        use arrow_array::Date32Array;
        let schema = opt_schema(vec![(1, "d", PrimitiveType::Date)]);
        let key_columns = vec![(1, "d".to_string(), PrimitiveType::Date)];
        let delete_rows = vec![vec![Some(Datum::date(100))]];
        let data: ArrayRef = Arc::new(Date32Array::from(vec![Some(100), Some(200)]));
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["d"], delete_rows, vec![("d", data)]);
        assert_eq!(mask, vec![true, false]);
    }

    /// Binary key — byte-string equality.
    #[test]
    fn test_h6_set_binary_matches_oracle() {
        use arrow_array::BinaryArray;
        let schema = opt_schema(vec![(1, "bin", PrimitiveType::Binary)]);
        let key_columns = vec![(1, "bin".to_string(), PrimitiveType::Binary)];
        let delete_rows = vec![vec![Some(Datum::binary(vec![1u8, 2, 3]))]];
        let data: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(&[1u8, 2, 3][..]),
            Some(&[9u8][..]),
        ]));
        let mask = assert_set_matches_oracle(schema, key_columns, &["bin"], delete_rows, vec![(
            "bin", data,
        )]);
        assert_eq!(mask, vec![true, false]);
    }

    /// Time key (Int64-backed temporal, micros from midnight) — the fast-path mask must equal the
    /// predicate oracle, proving the new `get_arrow_datum` Time arm and the re-admitted gate agree.
    /// Before this change a Time-keyed eq-delete errored `FeatureUnsupported` in the predicate path.
    #[test]
    fn test_h6_set_time_matches_oracle() {
        use arrow_array::Time64MicrosecondArray;
        let schema = opt_schema(vec![(1, "t", PrimitiveType::Time)]);
        let key_columns = vec![(1, "t".to_string(), PrimitiveType::Time)];
        // 01:01:01 = 3_661_000_000 micros; 12:00:00 = 43_200_000_000 micros.
        let delete_rows = vec![vec![Some(Datum::time_micros(3_661_000_000).unwrap())]];
        let data: ArrayRef = Arc::new(Time64MicrosecondArray::from(vec![
            Some(3_661_000_000i64),
            Some(43_200_000_000),
        ]));
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["t"], delete_rows, vec![("t", data)]);
        assert_eq!(mask, vec![true, false]);
    }

    /// Fixed(n) key (FixedSizeBinary(n), fixed-width byte string) — fast-path mask must equal the
    /// predicate oracle, proving the new `get_arrow_datum` Fixed arm and the re-admitted gate agree.
    /// Before this change a Fixed-keyed eq-delete errored `FeatureUnsupported` in the predicate path.
    #[test]
    fn test_h6_set_fixed_matches_oracle() {
        use arrow_array::FixedSizeBinaryArray;
        let schema = opt_schema(vec![(1, "f", PrimitiveType::Fixed(4))]);
        let key_columns = vec![(1, "f".to_string(), PrimitiveType::Fixed(4))];
        let delete_rows = vec![vec![Some(Datum::fixed(vec![0xDEu8, 0xAD, 0xBE, 0xEF]))]];
        let data: ArrayRef = Arc::new(
            FixedSizeBinaryArray::try_from_iter(
                vec![vec![0xDEu8, 0xAD, 0xBE, 0xEF], vec![
                    0x00u8, 0x01, 0x02, 0x03,
                ]]
                .into_iter(),
            )
            .expect("build Fixed(4) data column"),
        );
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["f"], delete_rows, vec![("f", data)]);
        assert_eq!(mask, vec![true, false]);
    }

    /// Uuid key — pins LE `UInt128` encode on the Datum side against Arrow FixedSizeBinary(16)
    /// via `Uuid::from_bytes`/`as_u128` on the probe side (critic-octo FK1 cycle 3 coverage).
    #[test]
    fn test_h6_set_uuid_matches_oracle() {
        use arrow_array::FixedSizeBinaryArray;
        use uuid::Uuid;

        let u_hit = Uuid::parse_str("00112233-4455-6677-8899-aabbccddeeff").expect("uuid");
        let u_miss = Uuid::parse_str("ffeeddcc-bbaa-9988-7766-554433221100").expect("uuid");
        let schema = opt_schema(vec![(1, "u", PrimitiveType::Uuid)]);
        let key_columns = vec![(1, "u".to_string(), PrimitiveType::Uuid)];
        let delete_rows = vec![vec![Some(Datum::uuid(u_hit))]];
        let data: ArrayRef = Arc::new(
            FixedSizeBinaryArray::try_from_iter(
                vec![u_hit.as_bytes().to_vec(), u_miss.as_bytes().to_vec()].into_iter(),
            )
            .expect("build Uuid data column"),
        );
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["u"], delete_rows, vec![("u", data)]);
        assert_eq!(mask, vec![true, false]);
    }

    /// THE KEY-NULL BAIL FOR THE NEW TYPES: a Time / Fixed batch with a NULL in the key column makes
    /// the fast path return `None`, and the predicate fallback — which previously ERRORED for
    /// Time/Fixed — SUCCEEDS (the `get_arrow_datum` arms) and, under Java nulls-first semantics
    /// (unit A2), KEEPS the NULL row: `survival(NULL) = (NULL != t) = TRUE`, matching Java's
    /// `StructLikeSet` eq-delete application (a null key cell equals no non-null delete value).
    /// This pins the (b)-leg of the gate admission: re-admitting Time/Fixed is sound only because
    /// the bail target no longer errors.
    #[test]
    fn test_h6_time_fixed_key_null_bails_to_predicate_without_error() {
        use arrow_array::{FixedSizeBinaryArray, Time64MicrosecondArray};

        // --- Time ---
        let schema = opt_schema(vec![(1, "t", PrimitiveType::Time)]);
        let key_columns = vec![(1, "t".to_string(), PrimitiveType::Time)];
        let delete_rows = vec![vec![Some(Datum::time_micros(3_661_000_000).unwrap())]];
        let set = EqDeleteKeySet::try_build(key_columns, delete_rows.clone())
            .expect("Time key column must build a set (now eligible)");
        let data: ArrayRef = Arc::new(Time64MicrosecondArray::from(vec![
            Some(3_661_000_000i64),
            Some(43_200_000_000),
            None, // key-column NULL → forces the predicate fallback
        ]));
        let batch = batch_with_field_ids(vec![("t", data)]);
        assert_eq!(
            set.delete_mask(&batch).expect("delete_mask"),
            None,
            "a key-column NULL must force the predicate fallback for Time"
        );
        // The predicate oracle for that batch must SUCCEED (no FeatureUnsupported) and KEEP the
        // NULL row: survival(NULL) = (NULL != t) = TRUE under Java nulls-first (null ≠ any
        // non-null delete value ⇒ not deleted, the Java StructLikeSet verdict).
        let oracle = multi_col_oracle_deleted_mask(&["t"], schema, &delete_rows, &batch);
        assert_eq!(
            oracle,
            vec![true, false, false],
            "predicate fallback must now evaluate a Time key (it errored before this change)"
        );

        // --- Fixed ---
        let schema = opt_schema(vec![(1, "f", PrimitiveType::Fixed(4))]);
        let key_columns = vec![(1, "f".to_string(), PrimitiveType::Fixed(4))];
        let delete_rows = vec![vec![Some(Datum::fixed(vec![0xDEu8, 0xAD, 0xBE, 0xEF]))]];
        let set = EqDeleteKeySet::try_build(key_columns, delete_rows.clone())
            .expect("Fixed key column must build a set (now eligible)");
        let data: ArrayRef = Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                vec![
                    Some(vec![0xDEu8, 0xAD, 0xBE, 0xEF]),
                    Some(vec![0x00u8, 0x01, 0x02, 0x03]),
                    None, // key-column NULL → forces the predicate fallback
                ]
                .into_iter(),
                4,
            )
            .expect("build Fixed(4) data column with a null"),
        );
        let batch = batch_with_field_ids(vec![("f", data)]);
        assert_eq!(
            set.delete_mask(&batch).expect("delete_mask"),
            None,
            "a key-column NULL must force the predicate fallback for Fixed"
        );
        let oracle = multi_col_oracle_deleted_mask(&["f"], schema, &delete_rows, &batch);
        assert_eq!(
            oracle,
            vec![true, false, false],
            "predicate fallback must now evaluate a Fixed key (it errored before this change)"
        );
    }

    /// MULTI-COLUMN key — membership on the full tuple == AND of per-column equality, with a partial
    /// match (one col matches, other doesn't → NOT deleted), a NULL DELETE cell (deletes nothing
    /// among non-null data), and a duplicate tuple. Data is non-null in both key columns → set path.
    #[test]
    fn test_h6_set_multi_column_matches_oracle() {
        use arrow_array::StringArray;
        let schema = opt_schema(vec![
            (1, "id", PrimitiveType::Long),
            (2, "name", PrimitiveType::String),
        ]);
        let key_columns = vec![
            (1, "id".to_string(), PrimitiveType::Long),
            (2, "name".to_string(), PrimitiveType::String),
        ];
        let delete_rows = vec![
            vec![Some(Datum::long(1)), Some(Datum::string("a"))],
            vec![Some(Datum::long(2)), None], // NULL in second cell — no non-null data matches it
            vec![Some(Datum::long(1)), Some(Datum::string("a"))], // duplicate
        ];
        let id: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(1i64),
            Some(1),
            Some(2),
            Some(2),
        ]));
        let name: ArrayRef = Arc::new(StringArray::from(vec![
            Some("a"), // (1,a) → deleted
            Some("b"), // (1,b) → partial, NOT deleted
            Some("y"), // (2,y) → NOT deleted (delete tuple 2 has NULL name)
            Some("x"), // (2,x) → NOT deleted
        ]));
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["id", "name"], delete_rows, vec![
                ("id", id),
                ("name", name),
            ]);
        assert_eq!(mask, vec![true, false, false, false]);
    }

    /// Empty delete set deletes nothing; none-match leaves everything (non-null data → set path).
    #[test]
    fn test_h6_set_empty_and_none_match() {
        let schema = opt_schema(vec![(1, "v", PrimitiveType::Long)]);
        let key_columns = vec![(1, "v".to_string(), PrimitiveType::Long)];
        // none-match: a delete value absent from the data.
        let delete_rows = vec![vec![Some(Datum::long(999))]];
        let data: ArrayRef = Arc::new(Int64Array::from(vec![Some(1i64), Some(2)]));
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["v"], delete_rows, vec![("v", data)]);
        assert_eq!(mask, vec![false, false]);

        // empty delete set: nothing is deleted (try_build with zero rows still gates by type).
        let empty =
            EqDeleteKeySet::try_build(vec![(1, "v".to_string(), PrimitiveType::Long)], vec![])
                .expect("eligible type builds even with zero rows");
        assert!(empty.is_empty());
        let batch = batch_with_field_ids(vec![(
            "v",
            Arc::new(Int64Array::from(vec![Some(1i64), Some(2)])) as ArrayRef,
        )]);
        assert_eq!(empty.delete_mask(&batch).unwrap(), Some(vec![false, false]));
    }

    /// THE NULL-DATA SOUNDNESS BOUNDARY: a batch with a NULL in a key column makes `delete_mask`
    /// return `None` (route this batch to the predicate fallback). Under Java nulls-first
    /// semantics (unit A2) the predicate path KEEPS such a NULL row unless a NULL delete tuple
    /// matches it — the Java `StructLikeSet` verdict. The bail stays mandatory (conservative:
    /// the predicate path is the oracle); extending the set path to null keys is a possible
    /// future optimization now that the two agree.
    ///
    /// **MUTATION (FK1 P0):** delete the `column.null_count() > 0 { return Ok(None) }` bail in
    /// `EqDeleteKeySet::delete_mask` → this test must go RED (returns `Some(...)` instead of
    /// `None`). Re-run at tip during critic-octo; a mutation that was RED three commits ago is
    /// not RED.
    #[test]
    fn test_h6_set_returns_none_when_key_column_has_null() {
        let schema = opt_schema(vec![(1, "v", PrimitiveType::Long)]);
        let key_columns = vec![(1, "v".to_string(), PrimitiveType::Long)];
        let delete_rows = vec![vec![Some(Datum::long(3))]]; // no NULL delete tuple
        let set =
            EqDeleteKeySet::try_build(key_columns, delete_rows.clone()).expect("Long set builds");

        // Data row 2 is NULL in the key column.
        let data: ArrayRef = Arc::new(Int64Array::from(vec![Some(3i64), Some(9), None]));
        let batch = batch_with_field_ids(vec![("v", data)]);

        // Fast path bails → None (must use the predicate path for this batch).
        assert_eq!(
            set.delete_mask(&batch).expect("delete_mask"),
            None,
            "a key-column NULL must force the predicate fallback"
        );

        // And the predicate oracle for that same batch KEEPS the NULL row (Java nulls-first):
        // survival(NULL) = (NULL != 3) = TRUE — no NULL delete tuple exists, so the null-key row
        // is not deleted (Java StructLikeSet equality: null equals only null).
        let oracle = multi_col_oracle_deleted_mask(&["v"], schema, &delete_rows, &batch);
        assert_eq!(
            oracle,
            vec![true, false, false],
            "the NULL key-column row survives a value-only delete set under Java nulls-first \
             semantics"
        );
    }

    /// Critic-octo FK1 cycle 1+2: I64 store drops null delete cells. Null-only Long delete files
    /// must (a) not report `is_empty()` (apply seam must not skip them), and (b) `delete_mask`
    /// null-bail so the predicate's `col IS NULL` leaf still applies.
    ///
    /// **MUTATION:** empty-before-null order in `delete_mask`, or treat null-only I64 as
    /// `is_empty()==true` without null-bail → this test goes RED.
    #[test]
    fn test_h6_set_null_only_i64_delete_bails_on_null_data() {
        let schema = opt_schema(vec![(1, "v", PrimitiveType::Long)]);
        let key_columns = vec![(1, "v".to_string(), PrimitiveType::Long)];
        // Only NULL delete keys — I64 store ends empty (nulls cannot be stored as i64).
        let delete_rows = vec![vec![None], vec![None]];
        let set =
            EqDeleteKeySet::try_build(key_columns, delete_rows.clone()).expect("Long set builds");
        assert!(
            !set.is_empty(),
            "null-only I64 deletes drop nulls from the store but must NOT report is_empty — \
             apply seams that skip empty sets would under-delete null data"
        );

        let data: ArrayRef = Arc::new(Int64Array::from(vec![Some(1i64), None, Some(2)]));
        let batch = batch_with_field_ids(vec![("v", data)]);

        assert_eq!(
            set.delete_mask(&batch).expect("delete_mask"),
            None,
            "null-only I64 delete file + null data batch must bail to the predicate path \
             (empty short-circuit must not run first)"
        );

        // Oracle: NULL delete deletes only the NULL data row.
        let oracle = multi_col_oracle_deleted_mask(&["v"], schema, &delete_rows, &batch);
        assert_eq!(
            oracle,
            vec![false, true, false],
            "predicate oracle: null-only delete set deletes null data rows only"
        );

        // Non-null data: null deletes never match values → nothing deleted.
        let non_null: ArrayRef = Arc::new(Int64Array::from(vec![Some(1i64), Some(2)]));
        let non_null_batch = batch_with_field_ids(vec![("v", non_null)]);
        assert_eq!(
            set.delete_mask(&non_null_batch).expect("delete_mask"),
            Some(vec![false, false]),
            "null-only I64 deletes delete nothing among fully non-null data"
        );
    }

    /// Critic-octo FK1 cycle 2: simulate the apply-seam keep-mask loop (reader
    /// `eq_delete_keep_mask`) over a null-only I64 set. Skipping `is_empty` sets without calling
    /// `delete_mask` would yield keep-all; the production loop must call `delete_mask` and fall
    /// back when it returns `None`.
    #[test]
    fn test_h6_apply_seam_null_only_i64_does_not_keep_all() {
        let key_columns = vec![(1, "v".to_string(), PrimitiveType::Long)];
        let delete_rows = vec![vec![None]];
        let set = EqDeleteKeySet::try_build(key_columns, delete_rows).expect("Long set builds");
        let sets = [set];

        let data: ArrayRef = Arc::new(Int64Array::from(vec![Some(1i64), None]));
        let batch = batch_with_field_ids(vec![("v", data)]);
        let num_rows = batch.num_rows();

        // Mirror reader::eq_delete_keep_mask (always call delete_mask; no empty-skip).
        let mut keep = vec![true; num_rows];
        let mut all_sets_safe = true;
        for set in &sets {
            match set.delete_mask(&batch).expect("delete_mask") {
                Some(deleted) => {
                    for (k, d) in keep.iter_mut().zip(deleted.iter()) {
                        *k &= !*d;
                    }
                }
                None => {
                    all_sets_safe = false;
                    break;
                }
            }
        }
        assert!(
            !all_sets_safe,
            "null-only I64 + null data must force predicate fallback at the apply seam"
        );
        // And is_empty must not invite a skip:
        assert!(!sets[0].is_empty());
    }

    /// THE GATE: Float / Double key columns must NOT build a set (route to the predicate fallback),
    /// and Decimal / Unknown are likewise excluded. This is what keeps the proven-divergent float
    /// case on the untouched predicate path. (Time and Fixed are NOT excluded — they gained a
    /// `get_arrow_datum` arm and their equality is integer-/byte-identical; see the eligible-type
    /// assertions below and `test_h6_set_time_matches_oracle` / `test_h6_set_fixed_matches_oracle`.)
    ///
    /// **MUTATION (FK1 P0):** admit `PrimitiveType::Float`/`Double` in
    /// `EqDeleteKeySet::is_eligible_type` → `try_build` returns `Some` for a Double key and
    /// `test_h6_naive_set_diverges_on_negative_zero` documents the semantic break. This gate test
    /// must go RED on the `is_none()` assertions. Float Java-Comparator hashing remains a named
    /// follow-up seed (not tonight).
    #[test]
    fn test_h6_gate_excludes_float_double_decimal_unknown() {
        assert!(!EqDeleteKeySet::is_eligible_type(&PrimitiveType::Float));
        assert!(!EqDeleteKeySet::is_eligible_type(&PrimitiveType::Double));
        assert!(!EqDeleteKeySet::is_eligible_type(&PrimitiveType::Decimal {
            precision: 10,
            scale: 2
        }));
        assert!(!EqDeleteKeySet::is_eligible_type(&PrimitiveType::Unknown));
        // Time and Fixed are now ADMITTED: `get_arrow_datum` evaluates them (so a key-null bail to the
        // predicate path succeeds rather than erroring) and their equality is integer- (Time, i64
        // micros) / byte- (Fixed, fixed-width bytes) identical under both the Arrow `eq` kernel and
        // `Datum` `Eq`.
        assert!(EqDeleteKeySet::is_eligible_type(&PrimitiveType::Time));
        assert!(EqDeleteKeySet::is_eligible_type(&PrimitiveType::Fixed(16)));
        // Eligible representatives.
        assert!(EqDeleteKeySet::is_eligible_type(&PrimitiveType::Long));
        assert!(EqDeleteKeySet::is_eligible_type(&PrimitiveType::String));

        // A Double key column → try_build returns None (no fast path).
        assert!(
            EqDeleteKeySet::try_build(vec![(1, "d".to_string(), PrimitiveType::Double)], vec![
                vec![Some(Datum::double(0.0))]
            ],)
            .is_none(),
            "Double key must not build a fast-path set"
        );
        // A MIXED key (one eligible, one float) → None: the whole file falls back.
        assert!(
            EqDeleteKeySet::try_build(
                vec![
                    (1, "id".to_string(), PrimitiveType::Long),
                    (2, "d".to_string(), PrimitiveType::Double),
                ],
                vec![vec![Some(Datum::long(1)), Some(Datum::double(0.0))]],
            )
            .is_none(),
            "a key with any float column must not build a fast-path set"
        );
    }

    /// THE FALLBACK STILL CORRECT: with the gate routing Double to the predicate path, the
    /// `-0.0`/`+0.0` case the naive set got wrong is handled correctly — only `+0.0` is deleted by a
    /// `+0.0` delete (total-ordering eq), proving the float fallback preserves the old behavior.
    #[test]
    fn test_h6_float_fallback_preserves_predicate_semantics() {
        let schema = double_schema();
        let delete_cells = vec![Some(Datum::double(0.0f64))];
        let data_vals = vec![Some(-0.0f64), Some(0.0f64), Some(1.0f64)];
        let batch = double_batch(&data_vals);

        // The predicate path (the route the gate forces for Double) deletes only +0.0.
        let oracle = oracle_deleted_mask("v", schema, &delete_cells, &batch);
        assert_eq!(
            oracle,
            vec![false, true, false],
            "Double fallback via the predicate path keeps -0.0 and deletes only +0.0"
        );

        // And the gate indeed refuses a Double set, so this case CANNOT take the fast path.
        assert!(
            EqDeleteKeySet::try_build(
                vec![(1, "v".to_string(), PrimitiveType::Double)],
                delete_cells.into_iter().map(|c| vec![c]).collect(),
            )
            .is_none()
        );
    }

    // -- positional_delete_keep_mask range-walk vs naive `!contains` byte-identity ----------------

    use crate::delete_vector::DeleteVector;

    /// Builds a [`DeleteVector`] from explicit positions (deterministic; no RNG/clock).
    fn dv_from(positions: &[u64]) -> DeleteVector {
        let mut dv = DeleteVector::new(roaring::RoaringTreemap::new());
        for &p in positions {
            dv.insert(p);
        }
        dv
    }

    /// The naive oracle: the exact mask the range-walk must reproduce byte-for-byte.
    fn naive_keep_mask(dv: &DeleteVector, base: u64, num_rows: usize) -> BooleanArray {
        BooleanArray::from(
            (0..num_rows)
                .map(|i| !dv.contains(base + i as u64))
                .collect::<Vec<bool>>(),
        )
    }

    /// Asserts the range-walk helper is byte-identical to the naive `!contains` probe for one shape.
    fn assert_equiv(positions: &[u64], base: u64, num_rows: usize, label: &str) {
        let dv = dv_from(positions);
        let fast = positional_delete_keep_mask(&dv, base, num_rows);
        let naive = naive_keep_mask(&dv, base, num_rows);
        assert_eq!(
            fast, naive,
            "range-walk mask diverged from naive !contains for case `{label}` \
             (positions={positions:?}, base={base}, num_rows={num_rows})"
        );
        assert_eq!(
            fast.len(),
            num_rows,
            "mask length must equal num_rows for case `{label}`"
        );
    }

    /// The 2^32 high-bits boundary — the roaring-treemap inner/outer split. A window straddling it
    /// exercises the trap that `advance_to` walks `outer` when `high_bits < hi`.
    const KEY_BOUNDARY: u64 = 1 << 32;

    #[test]
    fn test_positional_keep_mask_equivalence_explicit_shapes() {
        // Empty window: num_rows == 0 (helper returns an empty mask, never indexes).
        assert_equiv(&[], 0, 0, "empty-window-no-deletes");
        assert_equiv(&[5, 10], 0, 0, "empty-window-with-deletes");
        assert_equiv(&[5, 10], 7, 0, "empty-window-base-nonzero");

        // Zero deletes over a real window.
        assert_equiv(&[], 0, 16, "no-deletes-base0");
        assert_equiv(&[], 100, 16, "no-deletes-base100");

        // No rows deleted because every delete is out of the window.
        assert_equiv(&[0, 1, 2], 10, 5, "deletes-entirely-below-window");
        assert_equiv(&[20, 21, 22], 10, 5, "deletes-entirely-above-window");

        // All rows deleted (dense contiguous run exactly covering the window).
        assert_equiv(&[0, 1, 2, 3, 4, 5, 6, 7], 0, 8, "all-rows-deleted-base0");
        assert_equiv(
            &[10, 11, 12, 13, 14],
            10,
            5,
            "all-rows-deleted-base-nonzero",
        );

        // Sparse deletes inside the window.
        assert_equiv(&[2, 5, 9], 0, 12, "sparse-base0");
        assert_equiv(&[103, 107, 111], 100, 16, "sparse-base100");

        // Dense contiguous run inside a larger window (some survivors on each side).
        assert_equiv(&[4, 5, 6, 7, 8], 0, 16, "dense-run-interior-base0");
        assert_equiv(&[54, 55, 56, 57], 50, 20, "dense-run-interior-base50");

        // Window-edge deletes: exactly at base, exactly at base+num_rows-1 (last row), and exactly
        // at base+num_rows (one past — must NOT flip any row).
        assert_equiv(&[10], 10, 5, "delete-exactly-at-base");
        assert_equiv(&[14], 10, 5, "delete-exactly-at-last-row");
        assert_equiv(&[15], 10, 5, "delete-exactly-one-past-window-must-not-flip");
        assert_equiv(
            &[9, 10, 14, 15],
            10,
            5,
            "edges-combined-below-at-base-at-last-one-past",
        );

        // A primed cache value that is itself the first in-window delete (cached >= base): the
        // refresh-only-if-stale branch must KEEP it. base==first delete with nothing below.
        assert_equiv(&[10, 12], 10, 5, "primed-cache-is-first-in-window-delete");

        // Stale primed cache: a delete strictly below base must be skipped by advance_to + refresh.
        assert_equiv(&[3, 12, 13], 10, 5, "stale-primed-cache-below-window");

        // base == 0 with deletes only at and after 0 (prime yields 0, in-window, must be kept).
        assert_equiv(&[0, 3, 7], 0, 8, "base0-prime-zero-in-window");

        // ---- the 2^32 high-bits boundary ----
        // Window straddling the boundary: base just below 1<<32, spanning above it.
        assert_equiv(
            &[KEY_BOUNDARY - 2, KEY_BOUNDARY, KEY_BOUNDARY + 3],
            KEY_BOUNDARY - 4,
            8,
            "window-straddles-2^32-with-deletes-on-both-sides",
        );
        // Delete exactly AT the boundary, window starting below it.
        assert_equiv(
            &[KEY_BOUNDARY],
            KEY_BOUNDARY - 2,
            5,
            "delete-exactly-at-2^32-boundary",
        );
        // Entirely above the boundary (high_bits == 1) — exercises advance_to walking outer.
        assert_equiv(
            &[KEY_BOUNDARY + 5, KEY_BOUNDARY + 9],
            KEY_BOUNDARY + 2,
            12,
            "window-entirely-above-2^32",
        );
        // Stale primed cache below the boundary, real deletes above it (advance_to must walk outer
        // AND the refresh must drop the stale low-bits value).
        assert_equiv(
            &[7, KEY_BOUNDARY + 1, KEY_BOUNDARY + 2],
            KEY_BOUNDARY,
            5,
            "stale-cache-below-boundary-deletes-above",
        );

        // ---- GAP GROUPS (a high-bits group absent between two present groups) ----
        // The exact silent-corruption repro: group 0 = {KB-2}, group 2 = {0}; group 1 ABSENT.
        // base = 2*KB-2 with a 3-row window [2*KB-2, 2*KB+1). advance_to(base) hits hi=1 (the absent
        // group), so the outer walk overshoots to group 2; the FIXED iterator must leave group 2 at
        // its start so the in-window delete at 2*KB is still yielded. The old code consumed it →
        // mask wrongly all-true.
        assert_equiv(
            &[KEY_BOUNDARY - 2, 2 * KEY_BOUNDARY],
            2 * KEY_BOUNDARY - 2,
            3,
            "gap-group-repro-deleted-row-survives",
        );
        // Variant: in-window delete at the FIRST index of the window (overshoot lands exactly on it).
        assert_equiv(
            &[KEY_BOUNDARY - 2, 2 * KEY_BOUNDARY],
            2 * KEY_BOUNDARY,
            4,
            "gap-group-in-window-delete-at-index-0",
        );
        // Variant: in-window delete at a LATER index within the overshot group.
        assert_equiv(
            &[KEY_BOUNDARY - 2, 2 * KEY_BOUNDARY + 3],
            2 * KEY_BOUNDARY - 1,
            8,
            "gap-group-in-window-delete-at-later-index",
        );
        // Variant: MULTIPLE consecutive gap groups (groups 1 and 2 absent; below in group 0, in-window
        // in group 3). The outer walk must skip both gaps and not consume the in-window delete.
        assert_equiv(
            &[KEY_BOUNDARY - 1, 3 * KEY_BOUNDARY],
            3 * KEY_BOUNDARY - 1,
            3,
            "two-consecutive-gap-groups",
        );
        // Variant: base sits IN a gap group (group 1) with deletes BOTH below (groups 0/1-low) and
        // above (group 2), and the window straddles the gap into the present higher group. base is
        // placed just below group 2 so a small window reaches the higher group's deletes.
        assert_equiv(
            &[5, KEY_BOUNDARY - 3, 2 * KEY_BOUNDARY, 2 * KEY_BOUNDARY + 1],
            2 * KEY_BOUNDARY - 2,
            5, // window [2*KB-2, 2*KB+3): reaches 2*KB and 2*KB+1 in the present higher group
            "base-in-gap-group-deletes-below-and-above",
        );
        // Variant: gap group but window ends BEFORE the overshot group's delete (in-window mask must
        // stay all-true even though a higher delete exists past the window).
        assert_equiv(
            &[KEY_BOUNDARY - 2, 2 * KEY_BOUNDARY + 100],
            2 * KEY_BOUNDARY - 2,
            5,
            "gap-group-higher-delete-past-window-no-flip",
        );
    }

    #[test]
    fn test_positional_keep_mask_equivalence_generated() {
        // Deterministic LCG (Numerical Recipes constants) — reproducible, no clock/RNG dependency.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };

        // Sweep many (base, num_rows, delete-density) shapes, including ones crossing 2^32.
        let bases = [
            0u64,
            1,
            63,
            1000,
            KEY_BOUNDARY - 10,
            KEY_BOUNDARY,
            KEY_BOUNDARY + 100,
        ];
        let widths = [1usize, 2, 7, 64, 200];

        let mut checked = 0usize;
        for &base in &bases {
            for &num_rows in &widths {
                for density_sel in 0..4u64 {
                    // Generate deletes spread across [base-8, base+num_rows+8) so windows see
                    // below/in/above-window positions, plus occasional far-away deletes.
                    let span_lo = base.saturating_sub(8);
                    let span_hi = base.saturating_add(num_rows as u64).saturating_add(8);
                    let span = (span_hi - span_lo).max(1);
                    let count = match density_sel {
                        0 => 0,
                        1 => 1 + (next() % 3),
                        2 => 3 + (next() % 7),
                        _ => span / 2,
                    };
                    let mut positions: Vec<u64> =
                        (0..count).map(|_| span_lo + (next() % span)).collect();
                    // Occasionally inject a far-below and far-above delete.
                    if next() % 2 == 0 {
                        positions.push(span_hi.saturating_add(1000));
                    }
                    if base > 1000 && next() % 2 == 0 {
                        positions.push(base.saturating_sub(1000));
                    }
                    positions.sort_unstable();
                    positions.dedup();

                    assert_equiv(
                        &positions,
                        base,
                        num_rows,
                        &format!(
                            "generated[base={base},num_rows={num_rows},density={density_sel}]"
                        ),
                    );
                    checked += 1;
                }
            }
        }
        assert!(
            checked >= bases.len() * widths.len() * 4,
            "generator must have exercised every (base, width, density) combination"
        );
    }

    /// Claim `file_path` and hand back the loading guard, failing the test if the file was not
    /// claimable.
    fn claim_pos_del(filter: &DeleteFilter, file_path: &str) -> PosDelLoadGuard {
        match filter
            .try_start_pos_del_load(file_path)
            .expect("a fresh positional delete file must be claimable")
        {
            PosDelLoadAction::Load(guard) => guard,
            _ => panic!("a fresh positional delete file must be claimed, not waited on"),
        }
    }

    /// Arm a waiter on an in-flight positional-delete load, failing the test if the file is not
    /// currently claimed by someone else.
    fn arm_pos_del_waiter(filter: &DeleteFilter, file_path: &str) -> OwnedNotified {
        match filter
            .try_start_pos_del_load(file_path)
            .expect("a claimed positional delete file must not error at claim time")
        {
            PosDelLoadAction::WaitFor(notified) => notified,
            _ => panic!("a file already claimed by another task must make this caller wait"),
        }
    }

    /// Risk pinned (upstream apache/iceberg-rust#2859): the positional-delete waiter must ARM its
    /// notifier while [`DeleteFilter::try_start_pos_del_load`] still holds the state lock, so a
    /// `notify_waiters()` that fires before the waiter awaits still wakes it. `notify_waiters()`
    /// stores no permit, so this test's ordering — publish FIRST, await SECOND — only completes
    /// when the `Notified` already existed at publish time.
    ///
    /// MUTATION (semantic revert to the base contract: `PosDelLoadAction::WaitFor` carries a raw
    /// `Arc<Notify>` and the caller calls `.notified()` at the await site): the future is created
    /// after the publish, the wakeup is lost, and the timeout below fires (RED — verified on the
    /// pre-fix tree, `Elapsed(())`).
    #[tokio::test]
    async fn test_pos_del_waiter_is_armed_before_the_publisher_can_notify() {
        let filter = DeleteFilter::default();
        let guard = claim_pos_del(&filter, "pos-del.parquet");
        let notified = arm_pos_del_waiter(&filter, "pos-del.parquet");

        // Publish + `notify_waiters()` through the production publisher, BEFORE the waiter awaits.
        guard.publish_loaded();

        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            filter.wait_for_pos_del_load("pos-del.parquet", notified),
        )
        .await
        .expect("a notification fired after arming must wake the waiter, not be lost")
        .expect("a published load must resolve as loaded");

        assert!(
            matches!(
                filter
                    .try_start_pos_del_load("pos-del.parquet")
                    .expect("a loaded file must not error at claim time"),
                PosDelLoadAction::AlreadyLoaded
            ),
            "the published load must be visible to later callers as AlreadyLoaded"
        );
    }

    /// Risk pinned: a positional-delete loader that dies WITHOUT publishing — an early `?` on an
    /// unreadable or corrupt file, a sibling task's error tearing the shared load stream down, an
    /// unwind, or a runtime shutdown — must move the entry to the terminal [`PosDelState::Failed`]
    /// and STILL wake its waiters, so each gets a typed error inside a BOUNDED await. The claiming
    /// task is the sole writer for its file, so without that transition the entry stays `Loading`
    /// forever and every waiter parks on a notification that can never be sent.
    ///
    /// MUTATION: disarming the guard before it drops (`self.armed = false`, i.e. no `Failed`
    /// publish) leaves the entry `Loading`; the waiter never wakes and the timeout fires (RED —
    /// verified on the pre-fix tree, which has no `Failed` variant at all: `Elapsed(())`).
    #[tokio::test]
    async fn test_dead_pos_del_loader_yields_a_typed_error_not_a_hang() {
        let filter = DeleteFilter::default();
        let guard = claim_pos_del(&filter, "pos-del.parquet");
        let notified = arm_pos_del_waiter(&filter, "pos-del.parquet");

        // The loader dies without publishing; nothing else will ever touch this entry.
        drop(guard);

        let error = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            filter.wait_for_pos_del_load("pos-del.parquet", notified),
        )
        .await
        .expect("a dead loader must not strand the waiter")
        .expect_err("a dead loader must surface a typed error to the waiter");

        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert!(
            error.to_string().contains("pos-del.parquet"),
            "the error must name the delete file, got: {error}"
        );
    }

    /// Risk pinned: [`PosDelState::Failed`] is TERMINAL — a later caller must NOT be handed a fresh
    /// `Load` claim (which would lie about the file having no deletes if it, too, silently died) or
    /// an `AlreadyLoaded` (which would resurrect every row the file deletes). It gets the same
    /// typed error the waiters got, at claim time, without awaiting anything.
    ///
    /// MUTATION: mapping `PosDelState::Failed` to `PosDelLoadAction::AlreadyLoaded` in
    /// `try_start_pos_del_load` makes this test's `expect_err` trip (RED) — and would silently drop
    /// the file's deletes in production.
    #[test]
    fn test_claiming_a_pos_del_file_whose_loader_died_errors() {
        let filter = DeleteFilter::default();
        let guard = claim_pos_del(&filter, "pos-del.parquet");
        drop(guard);

        let error = filter
            .try_start_pos_del_load("pos-del.parquet")
            .expect_err("a terminally failed positional delete file must not be re-claimed");

        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert!(
            error.to_string().contains("pos-del.parquet"),
            "the error must name the delete file, got: {error}"
        );
    }

    /// Risk pinned (upstream apache/iceberg-rust#2859, the equality-delete half): the eq-delete
    /// waiter must ARM its notifier inside [`DeleteFilter::lookup_or_arm_eq_del`], while the read
    /// lock is held. Both eq-delete accessors go through that one seam, so this pins both. The
    /// publisher here is the production one (`spawn_publisher` on the claim's guard), driven to
    /// completion — asserted, not assumed — BEFORE the waiter awaits.
    ///
    /// MUTATION (semantic revert to the base contract: clone the `Arc<Notify>` out of the lock and
    /// call `.notified()` at the await site): the future is created after the publish, the wakeup
    /// is lost, and the timeout below fires (RED).
    #[tokio::test]
    async fn test_eq_del_waiter_is_armed_before_the_publisher_can_notify() {
        let filter = DeleteFilter::default();
        let guard = filter
            .try_start_eq_del_load("eq-del.parquet")
            .expect("a fresh eq-delete file must be claimable");

        // Arm through the production seam, exactly as both accessors do.
        let EqDelLookup::Wait(notified) =
            filter.lookup_or_arm_eq_del("eq-del.parquet", |predicate, _| predicate.clone())
        else {
            panic!("a claimed but unloaded eq-delete file must make the caller wait");
        };

        let predicate = Reference::new("id").equal_to(Datum::long(10));
        let (tx, rx) = tokio::sync::oneshot::channel();
        guard.spawn_publisher(rx);
        tx.send((predicate.clone(), None))
            .expect("the publisher task must still be listening");

        // Drive the publisher to completion BEFORE awaiting: the ordering under test is
        // "notify fires, THEN the waiter awaits".
        for _ in 0..64 {
            if !matches!(
                recover_poison(filter.state.read())
                    .equality_deletes
                    .get("eq-del.parquet"),
                Some(EqDelState::Loading(_))
            ) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            matches!(
                recover_poison(filter.state.read())
                    .equality_deletes
                    .get("eq-del.parquet"),
                Some(EqDelState::Loaded(_, _))
            ),
            "the publisher must have published before the waiter awaits, or this test proves nothing"
        );

        tokio::time::timeout(std::time::Duration::from_secs(5), notified)
            .await
            .expect("a notification fired after arming must wake the waiter, not be lost");

        assert_eq!(
            filter
                .get_equality_delete_predicate_for_delete_file_path("eq-del.parquet")
                .await,
            Some(predicate),
            "the woken waiter must read the published predicate"
        );
    }

    /// Risk pinned: the notifier a waiter arms on (the one installed in the state by
    /// [`DeleteFilter::try_start_eq_del_load`]) MUST be the notifier the publisher eventually
    /// fires. The base contract minted a SECOND notifier when the loader registered its receiver,
    /// replacing the state entry — so a waiter that armed on the claim's notifier in the window
    /// between the two calls was never woken by anything.
    ///
    /// MUTATION: making the guard carry a fresh `Arc::new(Notify::new())` instead of the notifier
    /// installed in the state (either at claim time or at registration time) loses this waiter's
    /// wakeup and the timeout below fires (RED).
    #[tokio::test]
    async fn test_eq_del_claim_notifier_is_the_one_the_publisher_fires() {
        let filter = DeleteFilter::default();
        let guard = filter
            .try_start_eq_del_load("eq-del.parquet")
            .expect("a fresh eq-delete file must be claimable");

        let EqDelLookup::Wait(notified) =
            filter.lookup_or_arm_eq_del("eq-del.parquet", |predicate, _| predicate.clone())
        else {
            panic!("a claimed but unloaded eq-delete file must make the caller wait");
        };

        // The loader dies before it ever registers a receiver: the guard's `Drop` must publish the
        // terminal state on the SAME notifier this waiter armed on.
        drop(guard);

        tokio::time::timeout(std::time::Duration::from_secs(5), notified)
            .await
            .expect("the claim's notifier must be the one the publisher fires");

        assert!(
            filter
                .get_equality_delete_predicate_for_delete_file_path("eq-del.parquet")
                .await
                .is_none(),
            "a terminally failed eq-delete load must read as absence, so the caller errors"
        );
    }

    /// Risk pinned (the zero-yield teardown probe): the eq-delete publisher future can be dropped
    /// BEFORE IT IS EVER POLLED — a runtime torn down between `spawn` and the first poll. A future
    /// dropped unpolled runs no local destructors, so a guard constructed *inside* the `async move`
    /// block would never exist and the entry would strand at `Loading` — with every waiter parked
    /// forever. The guard is therefore created by the claim and CAPTURED by the future.
    ///
    /// MUTATION: rebuilding the guard inside `spawn_publisher`'s `async move` block (from cloned
    /// `state` / `notify` handles) leaves this entry `Loading` and the timeout below fires (RED).
    #[tokio::test]
    async fn test_never_polled_eq_del_publisher_yields_absence_not_a_hang() {
        let filter = DeleteFilter::default();

        // Register the publisher on a runtime that is then DESTROYED with ZERO yields, so its
        // future is queued but never polled. Done on a separate thread because dropping a runtime
        // from inside an async context panics. The sender is kept ALIVE for the rest of the test,
        // so a dropped sender cannot be what resolves the entry.
        let filter_for_teardown = filter.clone();
        let _tx = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build the throwaway runtime that hosts the publisher task");
            let tx = runtime.block_on(async {
                let (tx, rx) =
                    tokio::sync::oneshot::channel::<(Predicate, Option<EqDeleteKeySet>)>();
                filter_for_teardown
                    .try_start_eq_del_load("eq-del.parquet")
                    .expect("a fresh eq-delete file must be claimable")
                    .spawn_publisher(rx);
                tx
            });
            drop(runtime);
            tx
        })
        .join()
        .expect("the runtime-teardown thread must not panic");

        let predicate = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            filter.get_equality_delete_predicate_for_delete_file_path("eq-del.parquet"),
        )
        .await
        .expect("a never-polled publisher must not hang the waiter");

        assert!(
            predicate.is_none(),
            "a never-polled publisher must leave the entry terminally failed, read as absence"
        );
    }
}
