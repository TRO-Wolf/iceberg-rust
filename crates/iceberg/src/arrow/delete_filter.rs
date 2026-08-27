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
    /// The resolved file: its survival [`Predicate`], which is always present, and the hashed
    /// [`EqDeleteKeySet`] accelerator when every key column is type-eligible.
    Loaded(Predicate, Option<EqDeleteKeySet>),
    /// The load failed terminally: the loader dropped the oneshot sender without sending. Any
    /// error or cancellation in the load, parse, or send window does that. Waiters MUST treat this
    /// as terminal and surface absence, so the caller raises a typed error. A re-wait on the
    /// notifier blocks the scan forever.
    Failed,
}

/// Load state of one positional delete file. Retrieval is synchronous, so a positional delete
/// must be fully loaded before the reader proceeds. An equality delete may resolve later.
#[derive(Debug)]
enum PosDelState {
    /// A task is loading the file. Other tasks wait on the notifier.
    Loading(Arc<Notify>),
    /// The file is loaded and merged into the delete vector map.
    Loaded,
    /// The claiming task died without publishing its delete vectors. Any error or cancellation in
    /// the claim, read, parse, or merge window does that. The claiming task is the sole writer and
    /// runs once, so the state can never advance on its own.
    Failed(String),
}

/// The memo key for one task-shaped positional-delete resolution: the task's data file path plus
/// the sorted, deduplicated claim keys of its positional delete sources. Two tasks with the same
/// data file and the same delete set share one merged vector. The shared-state analogue of Java's
/// per-task `DeleteFilter.deleteRowPositions` memo field.
type PosDelResolutionKey = (String, Vec<String>);

#[derive(Debug, Default)]
struct DeleteFileFilterState {
    /// The load cache: parsed positional-delete content, PER SOURCE. The key is the source's
    /// [`pos_del_claim_key`], and each value maps a DATA file path to the positions that source
    /// deletes from it. Java `BaseDeleteLoader.getOrReadPosDeletes` caches the same shape.
    pos_del_contributions: HashMap<String, Arc<HashMap<String, DeleteVector>>>,
    /// Memoized per-task merged vectors. See [`PosDelResolutionKey`]. An entry installs only once
    /// every claim key it depends on is present, and contributions are immutable, so a memoized
    /// union can never go stale.
    ///
    /// Frozen as [`Arc<DeleteVector>`], because nothing mutates a memoized vector after it
    /// publishes. The resolve path only reads it.
    resolved_pos_dels: HashMap<PosDelResolutionKey, Arc<DeleteVector>>,
    equality_deletes: HashMap<String, EqDelState>,
    positional_deletes: HashMap<String, PosDelState>,
}

/// The resolved merge-on-read deletes for a scan, and the logic to apply them to Arrow batches.
///
/// The engine-facing analogue of Java `org.apache.iceberg.data.DeleteFilter`. A query engine that
/// builds its own physical scan reuses this instead of reimplementing Iceberg's sequence-number,
/// DV-supersedes-position, and null-coercion rules. The typical loop, per [`FileScanTask`] from
/// [`TableScan::plan_files`](crate::scan::TableScan::plan_files):
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
/// An engine that folds deletes into its own pushdown can read
/// [`deleted_row_positions`](Self::deleted_row_positions) and
/// [`equality_delete_predicate`](Self::equality_delete_predicate) directly, and skip
/// [`apply`](Self::apply).
#[derive(Clone, Debug, Default)]
pub struct DeleteFilter {
    state: Arc<RwLock<DeleteFileFilterState>>,
}

/// What the caller must do after it tries to claim a positional delete file.
#[derive(Debug)]
pub(crate) enum PosDelLoadAction {
    /// The file is not loaded, the caller should load it. The guard carries the claim: publish it
    /// with [`PosDelLoadGuard::publish_loaded`] once the delete vectors are merged into the
    /// filter, or let it drop and every waiter gets a typed error instead of hanging.
    Load(PosDelLoadGuard),
    /// The file is already loaded, nothing to do.
    AlreadyLoaded,
    /// Another task is loading the file. The caller MUST pass this future to
    /// [`DeleteFilter::wait_for_pos_del_load`], because `get_delete_vector` is synchronous. The
    /// future is ARMED HERE, under the state lock.
    WaitFor(OwnedNotified),
}

/// Publishes the TERMINAL state of one positional-delete file's load and wakes its waiters.
/// [`DeleteFilter::try_start_pos_del_load`] arms it in the same critical section that installs
/// [`PosDelState::Loading`], so the claim never exists without its guard.
/// [`PosDelLoadGuard::publish_loaded`] disarms it on success.
pub(crate) struct PosDelLoadGuard {
    state: Arc<RwLock<DeleteFileFilterState>>,
    notify: Arc<Notify>,
    file_path: String,
    armed: bool,
    /// The cause recorded by [`PosDelLoadGuard::note_failure`], so waiters learn WHY the load
    /// died, not only THAT it did.
    failure_reason: Option<String>,
}

/// Renders the claim, not the guarded state. A `Debug` that read the state would `try_read` a lock
/// this guard's own `publish` may hold.
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
            // Recover a poisoned guard rather than cascade the panic. This task is the sole
            // writer for its file, and a stranded `Loading` hangs every waiter below.
            let mut state = recover_poison(self.state.write());
            state
                .positional_deletes
                .insert(self.file_path.clone(), terminal);
        }
        self.armed = false;
        self.notify.notify_waiters();
    }

    /// Marks this file fully loaded and wakes every waiter. Call it only AFTER the file's delete
    /// vectors merge into the filter. A woken waiter reads them synchronously, and would otherwise
    /// see an empty or partial result.
    pub(crate) fn publish_loaded(mut self) {
        self.publish(PosDelState::Loaded);
    }

    /// Records the error that is about to end this load, then hands it back for `?` propagation.
    ///
    /// Only the task holding the guard sees that error. Every other consumer reads the terminal
    /// state, so without this they learn only THAT the load died. Use it on every failure path
    /// that has a cause. The paths without one still publish `Failed` from `Drop`.
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

/// Publishes the TERMINAL state of one equality-delete file's load and wakes its waiters. The
/// counterpart of [`PosDelLoadGuard`], armed by [`DeleteFilter::try_start_eq_del_load`] in the same
/// critical section that installs [`EqDelState::Loading`]. It carries the notifier that claim
/// installed, so the notifier waiters arm on is the one that fires.
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

    /// Spawns the task that turns the loader's oneshot into this file's terminal state. The loader
    /// sends the parsed predicate once it reads the file. A sender dropped without sending, which
    /// any error or cancellation in the load window does, makes the await err and the entry move to
    /// [`EqDelState::Failed`].
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

/// The outcome of one read of an equality-delete entry. See
/// [`DeleteFilter::lookup_or_arm_eq_del`].
enum EqDelLookup<T> {
    /// The entry is terminal or unknown. This is the answer.
    Ready(Option<T>),
    /// The entry was still loading. Await this ALREADY-ARMED future, then read the state again.
    Wait(OwnedNotified),
}

/// Recovers a poisoned lock guard instead of cascading the panic to every later scan.
///
/// The guarded [`DeleteFileFilterState`] holds `HashMap`s whose critical sections only insert,
/// get, and clone. No re-entrant user code can tear a collection mid-mutation, so a guard a
/// panicked holder left behind still wraps a coherent state. Recovery keeps concurrent scans
/// alive. This is the crate's policy for the delete-path locks. See `arrow/reader.rs`.
fn recover_poison<G>(result: std::sync::LockResult<G>) -> G {
    result.unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The claim key under which one positional-delete SOURCE loads and installs its contribution.
///
/// | Source | Key |
/// |---|---|
/// | a parquet position-delete file | its own path |
/// | a deletion vector in a Puffin file | `{puffin path}@{offset}`, since one file holds many |
/// | anything else | `None` |
///
/// The loader and [`DeleteFilter::resolve_delete_vector`] share this one key, so the claim, the
/// install, and the application cannot drift apart and silently drop deletes.
///
/// A Puffin entry with a missing or negative `content_offset` also gives `None`. The loader fails
/// loud on that metadata, so such an entry can never have a contribution to resolve.
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
    /// The merged positional-delete vector for a scan task: the union of the contributions the
    /// task's OWN delete files make to its data file. See [`Self::resolve_delete_vector`].
    pub(crate) fn get_delete_vector(
        &self,
        file_scan_task: &FileScanTask,
    ) -> Option<Arc<DeleteVector>> {
        self.resolve_delete_vector(&file_scan_task.deletes, file_scan_task.data_file_path())
    }

    /// Resolves the positional deletes that a task's `deletes` apply to its `data_file_path`: the
    /// union, over the task's own positional sources only, of each source's contribution. # Notes
    /// This is Java's per-task scope. Java builds one `data.DeleteFilter` per task over
    /// `task.deletes()` alone, and `deletedRowPositions()` merges only those files' contributions
    /// into a fresh index.
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
                    // The loader fails loud on this shape before a contribution exists, so only
                    // contract misuse reaches it here.
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

        // Snapshot the contribution maps under the read lock and union outside it. Contributions
        // are immutable once installed, so the snapshot cannot go stale. Poison is recovered, not
        // swallowed as `None`: a `None` reads as "no positional deletes" and resurrects rows.
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
                        // Only a resolve before this source's load completed reaches here. Never
                        // memoize a union computed without every listed source.
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

        // OR by reference, with no roaring clone per contribution. Each one is frozen after
        // install, and the merge publishes once as an `Arc<DeleteVector>`.
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
            // A concurrent resolver may have memoized the same key while the union ran outside
            // the lock. Return THEIRS, so one task shape shares a single frozen Arc.
            let mut state = recover_poison(self.state.write());
            let entry = state
                .resolved_pos_dels
                .entry((data_file_path.to_string(), claim_keys))
                .or_insert_with(|| merged.clone());
            return Some(entry.clone());
        }
        Some(merged)
    }

    /// Claims an equality delete file for loading and returns the guard that publishes its
    /// terminal state. `None` means another task owns it, or it is already terminal.
    pub(crate) fn try_start_eq_del_load(&self, file_path: &str) -> Option<EqDelLoadGuard> {
        let mut state = recover_poison(self.state.write());

        // A terminal `Failed` is NOT re-claimed. It is cached for this filter's life, like
        // `Loaded`, so a woken waiter's re-read stays unambiguous. A re-claim could install a
        // fresh `Loading` under that waiter.
        if state.equality_deletes.contains_key(file_path) {
            return None;
        }

        // The guard carries THIS notifier, so the notifier waiters arm on is the one that fires.
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

    /// Tries to claim a positional delete file for loading. The action tells the caller to load the
    /// file, to wait for another task, or to do nothing. # Errors When a previous loader terminated
    /// without publishing.
    pub(crate) fn try_start_pos_del_load(&self, file_path: &str) -> Result<PosDelLoadAction> {
        let mut state = recover_poison(self.state.write());

        if let Some(existing) = state.positional_deletes.get(file_path) {
            match existing {
                PosDelState::Loaded => return Ok(PosDelLoadAction::AlreadyLoaded),
                // ARM HERE, under the lock. See `PosDelLoadAction::WaitFor`.
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

    /// Waits for another task's positional-delete load to reach a terminal state. `notified` MUST
    /// be the future [`Self::try_start_pos_del_load`] armed under the state lock. A `Notified`
    /// created here reopens the lost-wakeup window.
    pub(crate) async fn wait_for_pos_del_load(
        &self,
        file_path: &str,
        notified: OwnedNotified,
    ) -> Result<()> {
        notified.await;

        // The loading task publishes a TERMINAL state under the write lock and only then fires
        // the notifier, so a woken waiter always sees `Loaded` or `Failed`. Neither is replaced.
        // Anything else means the notifier fired without a terminal transition.
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

    /// Reads one equality-delete entry once. It answers outright when the entry is terminal, and
    /// otherwise ARMS the notifier. # Notes The arming MUST happen under the read lock, the same
    /// handshake [`PosDelLoadAction::WaitFor`] documents.
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

    /// The equality-delete predicate for one delete file path.
    pub(crate) async fn get_equality_delete_predicate_for_delete_file_path(
        &self,
        file_path: &str,
    ) -> Option<Predicate> {
        match self.lookup_or_arm_eq_del(file_path, |predicate, _| predicate.clone()) {
            EqDelLookup::Ready(predicate) => return predicate,
            EqDelLookup::Wait(notified) => notified.await,
        }

        // Once the notifier fires the entry is terminal, and neither state is ever replaced.
        // Read anything other than `Loaded` as absence, so the caller raises a typed error.
        match self.lookup_or_arm_eq_del(file_path, |predicate, _| predicate.clone()) {
            EqDelLookup::Ready(predicate) => predicate,
            EqDelLookup::Wait(_) => None,
        }
    }

    /// The hashed [`EqDeleteKeySet`] accelerator for an eq-delete file, awaiting its load.
    /// `Some(set)` means every key column is type-eligible, so the fast path applies. `None` means
    /// the caller uses the predicate path.
    pub(crate) async fn get_equality_delete_keyset_for_delete_file_path(
        &self,
        file_path: &str,
    ) -> Option<EqDeleteKeySet> {
        // A failed or unknown load gives no key set, which routes the task onto the predicate
        // path. That path raises the typed error. The outer `Option` is the entry's presence, and
        // the inner one is the file's fast-path eligibility.
        match self.lookup_or_arm_eq_del(file_path, |_, key_set| key_set.clone()) {
            EqDelLookup::Ready(key_set) => return key_set.flatten(),
            EqDelLookup::Wait(notified) => notified.await,
        }

        // After the notifier fires the entry is terminal. Anything other than `Loaded` gives
        // `None`, so the caller uses the predicate path.
        match self.lookup_or_arm_eq_del(file_path, |_, key_set| key_set.clone()) {
            EqDelLookup::Ready(key_set) => key_set.flatten(),
            EqDelLookup::Wait(_) => None,
        }
    }

    /// The routing gate for the fast path. It collects the hashed key sets for ALL of `task`'s
    /// equality-delete files. `Some(sets)` needs every file to be fast-path-eligible and to share
    /// one key-column schema, so their delete masks combine under one tuple shape. `None` sends
    /// the whole task to the predicate path.
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
            // One file without a key set disables the fast path for the task.
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

    /// Builds the combined equality-delete predicate for a task, bound to its schema.
    pub(crate) async fn build_equality_delete_predicate(
        &self,
        file_scan_task: &FileScanTask,
    ) -> Result<Option<BoundPredicate>> {
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

    /// Installs the parsed `data file path` to `positions` map of ONE freshly loaded positional
    /// delete source, under the claim the loading task holds. Call it BEFORE
    /// [`PosDelLoadGuard::publish_loaded`], in the same await-free block. A woken waiter resolves
    /// synchronously, so publishing first hands it an absent contribution.
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

/// The typed error every consumer of a terminally-failed positional-delete load receives. The
/// claim-time and post-wake paths must render one failure, so it lives in one place.
///
/// `reason` is the cause [`PosDelState::Failed`] carries. It renders inline rather than through
/// `with_source`, because it is a message the failing task left behind, not an [`Error`] this one
/// wraps. No source chain is dropped here.
fn pos_del_load_failed_error(file_path: &str, reason: &str) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        format!(
            "the loader for positional delete file '{file_path}' terminated without publishing \
             its deletes: {reason}"
        ),
    )
}

/// The stable engine-facing surface. It mirrors Java `org.apache.iceberg.data.DeleteFilter`.
impl DeleteFilter {
    /// Loads and resolves every merge-on-read delete that applies to `task`, through `file_io`.
    ///
    /// Run it beside your own data-file read. Position deletes and deletion vectors are resolved
    /// when it returns. An equality-delete predicate resolves on the first
    /// [`equality_delete_predicate`](Self::equality_delete_predicate) call.
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

    /// The merged bitmap of deleted 0-based positions in `task`'s data file, or `None`. Java
    /// `DeleteFilter.deletedRowPositions()`. It is populated once [`load`](Self::load) returns.
    pub fn deleted_row_positions(&self, task: &FileScanTask) -> Option<Arc<DeleteVector>> {
        self.get_delete_vector(task)
    }

    /// The combined equality-delete predicate for `task`, bound to its schema. A row SURVIVES when
    /// the predicate evaluates TRUE, because it negates the delete condition. `None` when the task
    /// has no equality deletes. Java `DeleteFilter.eqDeletedRowFilter()`.
    pub async fn equality_delete_predicate(
        &self,
        task: &FileScanTask,
    ) -> Result<Option<BoundPredicate>> {
        self.build_equality_delete_predicate(task).await
    }

    /// Applies `task`'s deletes to one Arrow `batch` of its data file and returns the surviving
    /// rows. Java `DeleteFilter.filter`. `row_base` is the 0-based position of `batch`'s first row
    /// in the data file, the `_pos` of row 0.
    pub fn apply(
        &self,
        task: &FileScanTask,
        batch: RecordBatch,
        row_base: u64,
        equality_predicate: Option<&BoundPredicate>,
    ) -> Result<RecordBatch> {
        let num_rows = batch.num_rows();

        // The memoized vector is frozen, so this apply is lock-free on the bitmap.
        let positional_mask: Option<BooleanArray> = match self.get_delete_vector(task) {
            Some(deletes) => {
                if deletes.is_empty() {
                    None
                } else {
                    // The range walk is byte-identical to a per-row `!contains` probe, and costs
                    // O(deletes in the window). See `positional_delete_keep_mask`.
                    Some(positional_delete_keep_mask(
                        deletes.as_ref(),
                        row_base,
                        num_rows,
                    ))
                }
            }
            None => None,
        };

        // The mask is already two-valued under Java nulls-first semantics, where a NULL key cell
        // survives a value delete. The coercion is defense in depth.
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

/// Coerces a three-valued keep-mask to two-valued, so every NULL drops its row. The Parquet
/// `RowFilter` never keeps a null result either. This is defense in depth:
/// `evaluate_predicate_to_mask` already returns two-valued masks.
fn coerce_nulls_to_false(mask: &BooleanArray) -> BooleanArray {
    if mask.null_count() == 0 {
        return mask.clone();
    }
    BooleanArray::from_iter((0..mask.len()).map(|i| Some(mask.is_valid(i) && mask.value(i))))
}

pub(crate) fn is_equality_delete(f: &FileScanTaskDeleteFile) -> bool {
    matches!(f.file_type, DataContentType::EqualityDeletes)
}

/// Builds a positional-delete keep-mask for the row window `[base, base + num_rows)`. Index `i` is
/// `false` when position `base + i` is deleted.
///
/// The result is byte-identical to a per-row `!deletes.contains(base + i)` probe. It range-walks
/// the ascending [`DeleteVectorIterator`] instead, which costs O(deletes in the window). The
/// Parquet path uses the same walk in `ArrowReader::build_deletes_row_selection`. This one serves
/// the Avro and ORC decode path, which applies deletes to an already-decoded batch.
///
/// # Notes
///
/// The prime, conditional `advance_to`, and refresh sequence below must not be reordered.
/// [`DeleteVectorIterator::advance_to`] has three edges:
///
/// | Edge | Consequence here |
/// |---|---|
/// | it does nothing until one `next()` primes the iterator | call `next()` once first |
/// | it cannot un-yield a value already pulled into `cached` | keep a primed `cached >= base` |
/// | it is a hint, and may leave the iterator below the window | the walk re-checks `pos >= base` |
///
/// `saturating_add` keeps a window abutting `u64::MAX` from wrapping. The `(pos - base) as usize`
/// index is bounded by `pos < end`, so the cast cannot truncate.
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
    // PRIME: advance_to does nothing until the iterator has yielded once.
    let mut cached = iter.next();
    // Skip past deletes below the window, but ONLY when the primed value predates it. That drives
    // advance_to strictly forward. An in-window primed value is the first real delete, and
    // advance_to cannot rewind, so leave it alone.
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
        // A residual `pos < base` is a below-window delete advance_to could not skip. Drop it and
        // keep walking, because the iterator ascends.
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
                first_row_id: None,
                file_sequence_number: None,
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
                first_row_id: None,
                file_sequence_number: None,
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
    /// existed must still deserialize. The new fields default, and the format defaults to Parquet.
    /// A breaking serde change here invalidates every previously serialized scan task.
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
            first_row_id: None,
            file_sequence_number: None,
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
        // BOUNDED: this call reaches the eq-delete wait path, so a lost-wakeup regression never
        // returns. Without the timeout that is a hung CI job instead of a red test.
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

    /// The public engine-facing surface: `DeleteFilter::load`, the position accessor, then `apply`
    /// on a batch the engine read itself.
    #[tokio::test]
    async fn test_public_delete_filter_load_and_apply() {
        use arrow_array::Array;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();
        let tasks = setup(table_location);

        // The public constructor resolves the task's deletes without the caching loader.
        let filter = DeleteFilter::load(&tasks[0], file_io).await.unwrap();

        // Data file 1 deletes {0,1,3,5,6,8,20,21,22,23,1022,1023}, so 12 distinct positions.
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

        // A 10-row batch covers positions 0..9. Deleted there: {0,1,3,5,6,8}. Survivors {2,4,7,9}.
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

    /// Risk pinned: a FAILED equality-delete load must give the waiting consumer a typed error
    /// inside a BOUNDED await, never a hang. A dropped oneshot sender moves the entry to
    /// `EqDelState::Failed` and STILL wakes the waiters.
    ///
    /// MUTATION: revert the terminal transition to `eq_del.await.unwrap()`. The entry stays
    /// `Loading` forever and this test times out (RED).
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
            first_row_id: None,
            file_sequence_number: None,
        };

        let filter = DeleteFilter::default();
        let (tx, rx) = tokio::sync::oneshot::channel::<(Predicate, Option<EqDeleteKeySet>)>();
        filter
            .try_start_eq_del_load("eq-del.parquet")
            .expect("a fresh eq-delete file must be claimable")
            .spawn_publisher(rx);

        // The loader fails after registration: nothing is sent and the sender drops, exactly as
        // an early return in the load window does.
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

    /// Builds a parquet positional-delete task entry. Metadata only, never opened.
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

    /// Risk pinned: a thread that panics while holding the `state` write guard poisons the
    /// `RwLock`, and later scan operations must RECOVER rather than cascade the panic.
    ///
    /// MUTATION: restore `self.state.write().unwrap()` in `install_pos_del_contribution` or
    /// `try_start_pos_del_load`. Both calls then panic on the poisoned lock (RED).
    #[test]
    fn test_poisoned_state_lock_recovers_instead_of_cascading() {
        let filter = DeleteFilter::default();
        // Claim BEFORE poisoning, so only the writer under test meets the poisoned lock.
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

        // `install_pos_del_contribution` must not panic, and its write must land.
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
        // Publishing the claim also meets the poisoned lock, through the guard's recover path.
        guard.publish_loaded();

        // `try_start_pos_del_load` must also recover and proceed.
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

    /// Risk pinned: a poisoned `state` lock must NOT make `resolve_delete_vector` return `None`
    /// for a present contribution. `apply` reads `None` as "no positional deletes", so a
    /// poison-induced `None` drops the file's deletes and resurrects rows.
    ///
    /// MUTATION: revert the resolver's state read to `self.state.read().ok()` with an early
    /// `None`. The poison is swallowed as `None` and the `expect` below trips (RED).
    #[test]
    fn test_get_delete_vector_survives_poisoned_lock() {
        let filter = DeleteFilter::default();

        // Install a contribution exactly as the production loader does, so a correct read
        // returns `Some`.
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

        // The resolver must RECOVER the poison and still return the delete vector. A `None` here
        // resurrects deleted row 7.
        let dv = filter
            .resolve_delete_vector(&[parquet_pos_del_entry("pos-del.parquet")], "data.parquet")
            .expect("a present delete vector must survive a poisoned state lock, not read as None");
        assert!(
            dv.contains(7),
            "the recovered delete vector must still carry its deleted positions"
        );
    }

    /// Memoized positional vectors freeze as `Arc<DeleteVector>` and are shared by pointer across
    /// resolvers of one task shape.
    ///
    /// MUTATION: re-wrap every resolve in a fresh `Arc::new`, either by skipping the memo install
    /// or by cloning the bitmap each time. `Arc::ptr_eq` goes RED.
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

    /// A multi-source resolve ORs by reference into one frozen Arc. Positions from BOTH sources
    /// must appear, and a second resolve of the same key shares the Arc.
    ///
    /// MUTATION: merge only the first contribution. The cardinality and contains asserts go RED.
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

    /// The equality path does not share the cross-task over-delete defect. Equality-delete state is
    /// a load cache keyed by DELETE FILE path, and application iterates `task.deletes`, so a
    /// predicate loaded for one task cannot fold into a task that does not list its file. Java
    /// `DeleteFilter.applyEqDeletes` reads only the task's own partitioned list.
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
            first_row_id: None,
            file_sequence_number: None,
        };

        let filter = DeleteFilter::default();
        // Load TWO eq-delete predicates into ONE shared filter, through the production machinery.
        for (path, value) in [("eq-del-1.parquet", 10i64), ("eq-del-2.parquet", 20i64)] {
            let (tx, rx) = tokio::sync::oneshot::channel();
            filter
                .try_start_eq_del_load(path)
                .expect("a fresh eq-delete file must be claimable")
                .spawn_publisher(rx);
            tx.send((Reference::new("id").equal_to(Datum::long(value)), None))
                .expect("the publisher task must be listening");
        }

        // The task lists ONLY eq-del-1, so its predicate must be exactly eq-del-1's.
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

        // A task listing NO eq deletes gets no predicate, whatever the shared state holds.
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

    // Equivalence harness: eq-delete SET membership against the production PREDICATE path.
    //
    // Production builds a leaf predicate per delete cell, AND-folds the cells, negates per row,
    // AND-folds the rows, binds, evaluates with the arrow kernels, and coerces NULL to `false`. A
    // row is DELETED when that makes the survival predicate FALSE.
    //
    // These tests pin the exact semantics a set-membership rewrite must reproduce, and show where
    // a naive `HashSet<Datum>` diverges from the oracle.

    /// The production deleted-mask oracle for a single-column eq-delete. It builds the survival
    /// predicate as `parse_equality_deletes_record_batch_stream` does, binds it, evaluates it over
    /// `data_batch`, coerces nulls to false, and returns `!survives`.
    fn oracle_deleted_mask(
        col_name: &str,
        schema: SchemaRef,
        delete_cells: &[Option<Datum>],
        data_batch: &RecordBatch,
    ) -> Vec<bool> {
        // The per-row survival predicate, exactly as production builds it.
        let mut row_predicates: Vec<Predicate> = Vec::new();
        for cell in delete_cells {
            let leaf = match cell {
                Some(datum) => Reference::new(col_name).equal_to(datum.clone()),
                None => Reference::new(col_name).is_null(),
            };
            row_predicates.push(leaf.not().rewrite_not());
        }
        // A balanced AND-fold, matching production's tree builder.
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

    /// The candidate set path for a SINGLE column. Each non-null delete value enters a
    /// `HashSet<Datum>`, and a null delete cell is remembered. A row is deleted when its value is
    /// in the set, or it is null and a null delete cell exists.
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

    /// The provably safe case: integers, with NULL deletes, NULL data rows, duplicate keys, an
    /// all-match, and a none-match. Integers have no NaN or signed-zero hazard, and the `col IS
    /// NULL` leaf coincides with set null handling, so the set path matches the oracle exactly.
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
        // The expected mask: 3 and 7 deleted, 9 kept, NULL deleted by the NULL delete, 100 kept.
        assert_eq!(oracle, vec![true, true, false, true, false]);
    }

    /// Divergence proof on signed zero. Production compares floats with `arrow_ord::cmp::eq`,
    /// whose kernels use TOTAL ordering, so `-0.0` and `+0.0` are DISTINCT and a `+0.0` delete
    /// spares `-0.0`. A `HashSet<Datum>` keyed on `OrderedFloat` collapses them into one key and
    /// deletes both. The masks differ on row 0, which is why a naive set rewrite is unsound.
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

        // The oracle deletes only +0.0. Under total ordering -0.0 is a distinct value.
        assert_eq!(
            oracle,
            vec![false, true, false],
            "total-ordering eq distinguishes -0.0 from +0.0: only +0.0 deleted"
        );
        // The naive set deletes both, because OrderedFloat collapses them. That is the divergence.
        assert_eq!(candidate, vec![true, true, false]);
        assert_ne!(
            oracle, candidate,
            "the naive HashSet<Datum> set path MUST diverge from the predicate oracle on signed \
             zero; this proves H6 cannot ship a naive set without matching arrow's total-ordering \
             float equality exactly"
        );
    }

    /// Equivalence on `NaN`. The total-ordering kernel makes `NaN == NaN` TRUE, so a `NaN` delete
    /// deletes a `NaN` row. `OrderedFloat` agrees, so both paths agree. Java `StructLikeSet` is
    /// bit-wise and differs, but these tests pin the current Rust path.
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

        // Both paths delete the NaN row, because total ordering makes NaN equal NaN.
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

    // The real `EqDeleteKeySet` fast path, proven byte-identical to the predicate ORACLE across
    // the non-float type matrix, and the type GATE proven to route Float and Double back to the
    // predicate path.
    //
    // Each test builds a batch and delete tuples, runs the predicate oracle and the production
    // `EqDeleteKeySet::delete_mask`, then asserts the masks are IDENTICAL. Both sides start from
    // the SAME `Datum`s and decode the data column the same way, so the only thing under test is
    // that `Datum` equality matches the Arrow `eq` kernel for the admitted types.

    /// The multi-column predicate oracle. A row is DELETED when it matches some delete tuple under
    /// the production survival predicate. It builds exactly what
    /// `parse_equality_deletes_record_batch_stream` builds for one file.
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

    /// Builds a `RecordBatch` whose columns carry `PARQUET_FIELD_ID_META_KEY`, so the predicate
    /// evaluator and `EqDeleteKeySet::delete_mask` resolve the same columns.
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

    /// Drives the equivalence for a batch with NO NULL in a key column. It asserts
    /// `delete_mask` returns a mask byte-identical to the oracle, and returns it for the caller.
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

    /// Long key, with duplicates, an all-match, a none-match, and a NULL delete tuple that deletes
    /// nothing among non-null rows. The data has no key-column NULL, so the fast path applies.
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

    /// Int key over the I64 store. It pins the i32 to i64 widen on both the build and the probe
    /// side, so the store stays oracle-identical for Int.
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

    /// Timestamp key in micros, over the I64 store path.
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

    /// Timestamptz in micros. The same physical I64 path as Timestamp, pinning the type arm.
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

    /// TimestampNs over the I64 store.
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

    /// The multi-column Bytes store keeps null tags, unlike I64. A null-only delete tuple must
    /// keep the set non-empty and bail, so the predicate `IS NULL` leaves still apply.
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
        // A full-null delete tuple. Bytes encodes TAG_NULL per cell.
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
        // Only the all-null row matches the all-null delete tuple.
        assert_eq!(oracle, vec![false, true]);
    }

    /// String key, with an empty string and a no-match. Non-null data takes the set path.
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

    /// Date key. It confirms an Int32-backed temporal type compares as its integer backing.
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

    /// Binary key, on byte-string equality.
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

    /// Time key, in micros from midnight. The fast-path mask must equal the oracle, which proves
    /// the `get_arrow_datum` Time arm and the gate agree. A Time-keyed eq-delete once errored
    /// `FeatureUnsupported` in the predicate path.
    #[test]
    fn test_h6_set_time_matches_oracle() {
        use arrow_array::Time64MicrosecondArray;
        let schema = opt_schema(vec![(1, "t", PrimitiveType::Time)]);
        let key_columns = vec![(1, "t".to_string(), PrimitiveType::Time)];
        // 01:01:01 is 3_661_000_000 micros, and 12:00:00 is 43_200_000_000 micros.
        let delete_rows = vec![vec![Some(Datum::time_micros(3_661_000_000).unwrap())]];
        let data: ArrayRef = Arc::new(Time64MicrosecondArray::from(vec![
            Some(3_661_000_000i64),
            Some(43_200_000_000),
        ]));
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["t"], delete_rows, vec![("t", data)]);
        assert_eq!(mask, vec![true, false]);
    }

    /// Fixed(n) key, a fixed-width byte string. The fast-path mask must equal the oracle, which
    /// proves the `get_arrow_datum` Fixed arm and the gate agree. A Fixed-keyed eq-delete once
    /// errored `FeatureUnsupported` in the predicate path.
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

    /// Uuid key. It pins the little-endian `UInt128` encode on the Datum side against Arrow
    /// FixedSizeBinary(16) on the probe side.
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

    /// The key-null bail for Time and Fixed. A NULL in the key column makes the fast path return
    /// `None`, and the predicate fallback, which once errored for these types, now SUCCEEDS. Under
    /// Java nulls-first semantics it KEEPS the NULL row, because `survival(NULL)` is TRUE and a
    /// null key cell equals no non-null delete value. Admitting these types is sound only because
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
        // The oracle must SUCCEED and KEEP the NULL row. Under Java nulls-first a null never
        // equals a non-null delete value, so the row is not deleted.
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

    /// Multi-column key. Tuple membership equals the AND of per-column equality. It covers a
    /// partial match, which is not deleted, a NULL delete cell, which deletes nothing among
    /// non-null data, and a duplicate tuple. Both key columns are non-null, so the set path runs.
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

    /// An empty delete set deletes nothing, and a none-match leaves every row.
    #[test]
    fn test_h6_set_empty_and_none_match() {
        let schema = opt_schema(vec![(1, "v", PrimitiveType::Long)]);
        let key_columns = vec![(1, "v".to_string(), PrimitiveType::Long)];
        // A none-match: the delete value is absent from the data.
        let delete_rows = vec![vec![Some(Datum::long(999))]];
        let data: ArrayRef = Arc::new(Int64Array::from(vec![Some(1i64), Some(2)]));
        let mask =
            assert_set_matches_oracle(schema, key_columns, &["v"], delete_rows, vec![("v", data)]);
        assert_eq!(mask, vec![false, false]);

        // An empty delete set deletes nothing. `try_build` still gates by type at zero rows.
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

    /// The null-data soundness boundary. A NULL in a key column makes `delete_mask` return `None`,
    /// which routes the batch to the predicate fallback. Under Java nulls-first the predicate path
    /// KEEPS such a row unless a NULL delete tuple matches it.
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

        // The oracle KEEPS the NULL row. No NULL delete tuple exists, and under Java nulls-first
        // a null key equals only a null.
        let oracle = multi_col_oracle_deleted_mask(&["v"], schema, &delete_rows, &batch);
        assert_eq!(
            oracle,
            vec![true, false, false],
            "the NULL key-column row survives a value-only delete set under Java nulls-first \
             semantics"
        );
    }

    /// The I64 store drops null delete cells. A null-only Long delete file must not report
    /// `is_empty()`, so the apply seam does not skip it, and `delete_mask` must null-bail so the
    /// predicate's `col IS NULL` leaf still applies.
    ///
    /// MUTATION: check empty before null in `delete_mask`, or treat a null-only I64 set as empty
    /// without the null bail. This test goes RED.
    #[test]
    fn test_h6_set_null_only_i64_delete_bails_on_null_data() {
        let schema = opt_schema(vec![(1, "v", PrimitiveType::Long)]);
        let key_columns = vec![(1, "v".to_string(), PrimitiveType::Long)];
        // Only NULL delete keys, so the I64 store ends empty.
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

        // The oracle deletes only the NULL data row.
        let oracle = multi_col_oracle_deleted_mask(&["v"], schema, &delete_rows, &batch);
        assert_eq!(
            oracle,
            vec![false, true, false],
            "predicate oracle: null-only delete set deletes null data rows only"
        );

        // With non-null data a null delete matches nothing.
        let non_null: ArrayRef = Arc::new(Int64Array::from(vec![Some(1i64), Some(2)]));
        let non_null_batch = batch_with_field_ids(vec![("v", non_null)]);
        assert_eq!(
            set.delete_mask(&non_null_batch).expect("delete_mask"),
            Some(vec![false, false]),
            "null-only I64 deletes delete nothing among fully non-null data"
        );
    }

    /// Simulates the reader's `eq_delete_keep_mask` loop over a null-only I64 set. Skipping an
    /// `is_empty` set without calling `delete_mask` keeps every row. The production loop must call
    /// `delete_mask` and fall back when it returns `None`.
    #[test]
    fn test_h6_apply_seam_null_only_i64_does_not_keep_all() {
        let key_columns = vec![(1, "v".to_string(), PrimitiveType::Long)];
        let delete_rows = vec![vec![None]];
        let set = EqDeleteKeySet::try_build(key_columns, delete_rows).expect("Long set builds");
        let sets = [set];

        let data: ArrayRef = Arc::new(Int64Array::from(vec![Some(1i64), None]));
        let batch = batch_with_field_ids(vec![("v", data)]);
        let num_rows = batch.num_rows();

        // Mirror reader::eq_delete_keep_mask: always call delete_mask, and never skip on empty.
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
        // is_empty must not invite a skip.
        assert!(!sets[0].is_empty());
    }

    /// The type gate. Float, Double, Decimal, and Unknown key columns must NOT build a set, which
    /// keeps the divergent float case on the predicate path. Time and Fixed are admitted, because
    /// their equality is integer- or byte-identical.
    ///
    /// MUTATION: admit `PrimitiveType::Float` or `Double` in `EqDeleteKeySet::is_eligible_type`.
    /// `try_build` returns `Some` for a Double key and the `is_none()` assertions go RED.
    #[test]
    fn test_h6_gate_excludes_float_double_decimal_unknown() {
        assert!(!EqDeleteKeySet::is_eligible_type(&PrimitiveType::Float));
        assert!(!EqDeleteKeySet::is_eligible_type(&PrimitiveType::Double));
        assert!(!EqDeleteKeySet::is_eligible_type(&PrimitiveType::Decimal {
            precision: 10,
            scale: 2
        }));
        assert!(!EqDeleteKeySet::is_eligible_type(&PrimitiveType::Unknown));
        // Time and Fixed are ADMITTED. `get_arrow_datum` evaluates them, so a key-null bail to
        // the predicate path succeeds, and their equality is integer- or byte-identical under both
        // the Arrow `eq` kernel and `Datum` `Eq`.
        assert!(EqDeleteKeySet::is_eligible_type(&PrimitiveType::Time));
        assert!(EqDeleteKeySet::is_eligible_type(&PrimitiveType::Fixed(16)));
        // Eligible representatives.
        assert!(EqDeleteKeySet::is_eligible_type(&PrimitiveType::Long));
        assert!(EqDeleteKeySet::is_eligible_type(&PrimitiveType::String));

        // A Double key column gives None, so there is no fast path.
        assert!(
            EqDeleteKeySet::try_build(vec![(1, "d".to_string(), PrimitiveType::Double)], vec![
                vec![Some(Datum::double(0.0))]
            ],)
            .is_none(),
            "Double key must not build a fast-path set"
        );
        // A MIXED key gives None, so the whole file falls back.
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

    /// The fallback stays correct. With the gate routing Double to the predicate path, a `+0.0`
    /// delete deletes only `+0.0`, which is the case the naive set got wrong.
    #[test]
    fn test_h6_float_fallback_preserves_predicate_semantics() {
        let schema = double_schema();
        let delete_cells = vec![Some(Datum::double(0.0f64))];
        let data_vals = vec![Some(-0.0f64), Some(0.0f64), Some(1.0f64)];
        let batch = double_batch(&data_vals);

        // The predicate path, which the gate forces for Double, deletes only +0.0.
        let oracle = oracle_deleted_mask("v", schema, &delete_cells, &batch);
        assert_eq!(
            oracle,
            vec![false, true, false],
            "Double fallback via the predicate path keeps -0.0 and deletes only +0.0"
        );

        // The gate refuses a Double set, so this case CANNOT take the fast path.
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

    /// Builds a [`DeleteVector`] from explicit positions. Deterministic, with no RNG or clock.
    fn dv_from(positions: &[u64]) -> DeleteVector {
        let mut dv = DeleteVector::new(roaring::RoaringTreemap::new());
        for &p in positions {
            dv.insert(p);
        }
        dv
    }

    /// The naive oracle. The range walk must reproduce this mask byte for byte.
    fn naive_keep_mask(dv: &DeleteVector, base: u64, num_rows: usize) -> BooleanArray {
        BooleanArray::from(
            (0..num_rows)
                .map(|i| !dv.contains(base + i as u64))
                .collect::<Vec<bool>>(),
        )
    }

    /// Asserts the range walk is byte-identical to the naive `!contains` probe for one shape.
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

    /// The 2^32 high-bits boundary, where the roaring treemap splits inner from outer. A window
    /// across it exercises `advance_to` walking `outer` when `high_bits < hi`.
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

        // Window-edge deletes at base, at the last row, and one past the window, which must flip
        // no row.
        assert_equiv(&[10], 10, 5, "delete-exactly-at-base");
        assert_equiv(&[14], 10, 5, "delete-exactly-at-last-row");
        assert_equiv(&[15], 10, 5, "delete-exactly-one-past-window-must-not-flip");
        assert_equiv(
            &[9, 10, 14, 15],
            10,
            5,
            "edges-combined-below-at-base-at-last-one-past",
        );

        // A primed cache value that is itself the first in-window delete. The refresh-if-stale
        // branch must KEEP it.
        assert_equiv(&[10, 12], 10, 5, "primed-cache-is-first-in-window-delete");

        // A stale primed cache: a delete below base must be skipped by advance_to and refresh.
        assert_equiv(&[3, 12, 13], 10, 5, "stale-primed-cache-below-window");

        // base == 0 with deletes at and after 0. The prime yields an in-window 0, which must stay.
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
        // Entirely above the boundary, which exercises advance_to walking outer.
        assert_equiv(
            &[KEY_BOUNDARY + 5, KEY_BOUNDARY + 9],
            KEY_BOUNDARY + 2,
            12,
            "window-entirely-above-2^32",
        );
        // A stale primed cache below the boundary with real deletes above it. advance_to must walk
        // outer, and the refresh must drop the stale low-bits value.
        assert_equiv(
            &[7, KEY_BOUNDARY + 1, KEY_BOUNDARY + 2],
            KEY_BOUNDARY,
            5,
            "stale-cache-below-boundary-deletes-above",
        );

        // ---- GAP GROUPS (a high-bits group absent between two present groups) ----
        // The silent-corruption repro. Group 1 is ABSENT, so advance_to overshoots into group 2.
        // The iterator must leave group 2 at its start, so the in-window delete at 2*KB is still
        // yielded. The old code consumed it and the mask came back all-true.
        assert_equiv(
            &[KEY_BOUNDARY - 2, 2 * KEY_BOUNDARY],
            2 * KEY_BOUNDARY - 2,
            3,
            "gap-group-repro-deleted-row-survives",
        );
        // The in-window delete sits at the FIRST index, where the overshoot lands exactly on it.
        assert_equiv(
            &[KEY_BOUNDARY - 2, 2 * KEY_BOUNDARY],
            2 * KEY_BOUNDARY,
            4,
            "gap-group-in-window-delete-at-index-0",
        );
        // The in-window delete sits at a LATER index within the overshot group.
        assert_equiv(
            &[KEY_BOUNDARY - 2, 2 * KEY_BOUNDARY + 3],
            2 * KEY_BOUNDARY - 1,
            8,
            "gap-group-in-window-delete-at-later-index",
        );
        // MULTIPLE consecutive gap groups. The outer walk must skip both and keep the in-window
        // delete in group 3.
        assert_equiv(
            &[KEY_BOUNDARY - 1, 3 * KEY_BOUNDARY],
            3 * KEY_BOUNDARY - 1,
            3,
            "two-consecutive-gap-groups",
        );
        // base sits IN a gap group, with deletes both below and above, and the window straddles
        // the gap into the higher group.
        assert_equiv(
            &[5, KEY_BOUNDARY - 3, 2 * KEY_BOUNDARY, 2 * KEY_BOUNDARY + 1],
            2 * KEY_BOUNDARY - 2,
            5, // window [2*KB-2, 2*KB+3): reaches 2*KB and 2*KB+1 in the present higher group
            "base-in-gap-group-deletes-below-and-above",
        );
        // A gap group where the window ends BEFORE the overshot group's delete. The mask must stay
        // all-true.
        assert_equiv(
            &[KEY_BOUNDARY - 2, 2 * KEY_BOUNDARY + 100],
            2 * KEY_BOUNDARY - 2,
            5,
            "gap-group-higher-delete-past-window-no-flip",
        );
    }

    #[test]
    fn test_positional_keep_mask_equivalence_generated() {
        // A deterministic LCG, so the sweep is reproducible without a clock or RNG.
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
                    // Spread deletes across [base-8, base+num_rows+8), so a window sees positions
                    // below it, inside it, and above it.
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
                    // Sometimes inject a far-below and a far-above delete.
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

    /// Claims `file_path` and returns the loading guard, failing the test if it is not claimable.
    fn claim_pos_del(filter: &DeleteFilter, file_path: &str) -> PosDelLoadGuard {
        match filter
            .try_start_pos_del_load(file_path)
            .expect("a fresh positional delete file must be claimable")
        {
            PosDelLoadAction::Load(guard) => guard,
            _ => panic!("a fresh positional delete file must be claimed, not waited on"),
        }
    }

    /// Arms a waiter on an in-flight load, failing the test if nobody else holds the claim.
    fn arm_pos_del_waiter(filter: &DeleteFilter, file_path: &str) -> OwnedNotified {
        match filter
            .try_start_pos_del_load(file_path)
            .expect("a claimed positional delete file must not error at claim time")
        {
            PosDelLoadAction::WaitFor(notified) => notified,
            _ => panic!("a file already claimed by another task must make this caller wait"),
        }
    }

    /// Risk pinned: the positional-delete waiter must ARM its notifier while
    /// [`DeleteFilter::try_start_pos_del_load`] still holds the state lock. `notify_waiters()`
    /// stores no permit, so this test's publish-first, await-second ordering completes only when
    /// the `Notified` already existed at publish time. MUTATION: make `PosDelLoadAction::WaitFor`
    /// carry a raw `Arc<Notify>` and call `.notified()` at the await site.
    #[tokio::test]
    async fn test_pos_del_waiter_is_armed_before_the_publisher_can_notify() {
        let filter = DeleteFilter::default();
        let guard = claim_pos_del(&filter, "pos-del.parquet");
        let notified = arm_pos_del_waiter(&filter, "pos-del.parquet");

        // Publish and notify through the production publisher, BEFORE the waiter awaits.
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

    /// Risk pinned: a loader that dies WITHOUT publishing must move the entry to
    /// [`PosDelState::Failed`] and STILL wake its waiters, so each gets a typed error inside a
    /// BOUNDED await. The claiming task is the sole writer, so without that transition the entry
    /// stays `Loading` and every waiter parks on a notification nobody can send. MUTATION: disarm
    /// the guard before it drops, so no `Failed` publishes.
    #[tokio::test]
    async fn test_dead_pos_del_loader_yields_a_typed_error_not_a_hang() {
        let filter = DeleteFilter::default();
        let guard = claim_pos_del(&filter, "pos-del.parquet");
        let notified = arm_pos_del_waiter(&filter, "pos-del.parquet");

        // The loader dies without publishing, and nothing else touches this entry.
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

    /// Risk pinned: [`PosDelState::Failed`] is TERMINAL. A later caller must get neither a fresh
    /// `Load` claim, which would lie if it also died, nor an `AlreadyLoaded`, which would resurrect
    /// every row the file deletes. It gets the waiters' typed error at claim time.
    ///
    /// MUTATION: map `PosDelState::Failed` to `PosDelLoadAction::AlreadyLoaded`. The `expect_err`
    /// below trips (RED), and production silently drops the file's deletes.
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

    /// Risk pinned: the eq-delete waiter must ARM its notifier inside
    /// [`DeleteFilter::lookup_or_arm_eq_del`], while the read lock is held. Both eq-delete
    /// accessors go through that seam, so this pins both. The production publisher runs to
    /// completion, asserted rather than assumed, BEFORE the waiter awaits.
    #[tokio::test]
    async fn test_eq_del_waiter_is_armed_before_the_publisher_can_notify() {
        let filter = DeleteFilter::default();
        let guard = filter
            .try_start_eq_del_load("eq-del.parquet")
            .expect("a fresh eq-delete file must be claimable");

        // Arm through the production seam, as both accessors do.
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

        // Drive the publisher to completion BEFORE awaiting. The ordering under test is: notify
        // fires, then the waiter awaits.
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

    /// Risk pinned: the notifier a waiter arms on MUST be the notifier the publisher fires. The
    /// base contract minted a SECOND notifier at registration and replaced the state entry, so a
    /// waiter that armed on the claim's notifier in between was never woken.
    ///
    /// MUTATION: make the guard carry a fresh `Notify` instead of the installed one. This waiter's
    /// wakeup is lost and the timeout fires (RED).
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

        // The loader dies before it registers a receiver, so the guard's `Drop` must publish on
        // the SAME notifier this waiter armed on.
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

    /// Risk pinned: the eq-delete publisher future can be dropped BEFORE its first poll, when a
    /// runtime is torn down between `spawn` and that poll. Such a future runs no local destructors,
    /// so a guard built inside the `async move` block would never exist and the entry would strand
    /// at `Loading`. The claim therefore builds the guard and the future captures it.
    #[tokio::test]
    async fn test_never_polled_eq_del_publisher_yields_absence_not_a_hang() {
        let filter = DeleteFilter::default();

        // Register the publisher on a runtime that is then DESTROYED with ZERO yields, so its
        // future is queued and never polled. A separate thread is needed, because dropping a
        // runtime inside an async context panics. The sender stays ALIVE, so a dropped sender
        // cannot be what resolves the entry.
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
