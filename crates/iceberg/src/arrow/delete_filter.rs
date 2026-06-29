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
use std::sync::{Arc, Mutex, RwLock};

use arrow_arith::boolean::and;
use arrow_array::{Array, BooleanArray, RecordBatch};
use arrow_select::filter::filter_record_batch;
use tokio::sync::Notify;
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
}

#[derive(Debug, Default)]
struct DeleteFileFilterState {
    delete_vectors: HashMap<String, Arc<Mutex<DeleteVector>>>,
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
pub(crate) enum PosDelLoadAction {
    /// The file is not loaded, the caller should load it.
    Load,
    /// The file is already loaded, nothing to do.
    AlreadyLoaded,
    /// The file is currently being loaded by another task.
    /// The caller *must* wait for this notifier to ensure data availability
    /// before returning, as subsequent access (get_delete_vector) is synchronous.
    WaitFor(Arc<Notify>),
}

impl DeleteFilter {
    /// Retrieve a delete vector for the data file associated with a given file scan task
    pub(crate) fn get_delete_vector(
        &self,
        file_scan_task: &FileScanTask,
    ) -> Option<Arc<Mutex<DeleteVector>>> {
        self.get_delete_vector_for_path(file_scan_task.data_file_path())
    }

    /// Retrieve a delete vector for a data file
    pub(crate) fn get_delete_vector_for_path(
        &self,
        data_file_path: &str,
    ) -> Option<Arc<Mutex<DeleteVector>>> {
        self.state
            .read()
            .ok()
            .and_then(|st| st.delete_vectors.get(data_file_path).cloned())
    }

    pub(crate) fn try_start_eq_del_load(&self, file_path: &str) -> Option<Arc<Notify>> {
        let mut state = self.state.write().unwrap();

        // Skip if already loaded/loading - another task owns it
        if state.equality_deletes.contains_key(file_path) {
            return None;
        }

        // Mark as loading to prevent duplicate work
        let notifier = Arc::new(Notify::new());
        state
            .equality_deletes
            .insert(file_path.to_string(), EqDelState::Loading(notifier.clone()));

        Some(notifier)
    }

    /// Attempts to mark a positional delete file as "loading".
    ///
    /// Returns an action dictating whether the caller should load the file,
    /// wait for another task to load it, or do nothing.
    pub(crate) fn try_start_pos_del_load(&self, file_path: &str) -> PosDelLoadAction {
        let mut state = self.state.write().unwrap();

        if let Some(state) = state.positional_deletes.get(file_path) {
            match state {
                PosDelState::Loaded => return PosDelLoadAction::AlreadyLoaded,
                PosDelState::Loading(notify) => return PosDelLoadAction::WaitFor(notify.clone()),
            }
        }

        let notifier = Arc::new(Notify::new());
        state
            .positional_deletes
            .insert(file_path.to_string(), PosDelState::Loading(notifier));

        PosDelLoadAction::Load
    }

    /// Marks a positional delete file as successfully loaded and notifies any waiting tasks.
    pub(crate) fn finish_pos_del_load(&self, file_path: &str) {
        let notify = {
            let mut state = self.state.write().unwrap();
            if let Some(PosDelState::Loading(notify)) = state
                .positional_deletes
                .insert(file_path.to_string(), PosDelState::Loaded)
            {
                Some(notify)
            } else {
                None
            }
        };

        if let Some(notify) = notify {
            notify.notify_waiters();
        }
    }

    /// Retrieve the equality delete predicate for a given eq delete file path
    pub(crate) async fn get_equality_delete_predicate_for_delete_file_path(
        &self,
        file_path: &str,
    ) -> Option<Predicate> {
        let notifier = {
            match self.state.read().unwrap().equality_deletes.get(file_path) {
                None => return None,
                Some(EqDelState::Loading(notifier)) => notifier.clone(),
                Some(EqDelState::Loaded(predicate, _)) => {
                    return Some(predicate.clone());
                }
            }
        };

        notifier.notified().await;

        match self.state.read().unwrap().equality_deletes.get(file_path) {
            Some(EqDelState::Loaded(predicate, _)) => Some(predicate.clone()),
            _ => unreachable!("Cannot be any other state than loaded"),
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
        let notifier = {
            match self.state.read().unwrap().equality_deletes.get(file_path) {
                None => return None,
                Some(EqDelState::Loading(notifier)) => notifier.clone(),
                Some(EqDelState::Loaded(_, key_set)) => {
                    return key_set.clone();
                }
            }
        };

        notifier.notified().await;

        match self.state.read().unwrap().equality_deletes.get(file_path) {
            Some(EqDelState::Loaded(_, key_set)) => key_set.clone(),
            _ => unreachable!("Cannot be any other state than loaded"),
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
        for delete in &task.deletes {
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
        for delete in &file_scan_task.deletes {
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

    pub(crate) fn upsert_delete_vector(
        &mut self,
        data_file_path: String,
        delete_vector: DeleteVector,
    ) {
        let mut state = self.state.write().unwrap();

        let Some(entry) = state.delete_vectors.get_mut(&data_file_path) else {
            state
                .delete_vectors
                .insert(data_file_path, Arc::new(Mutex::new(delete_vector)));
            return;
        };

        *entry.lock().unwrap() |= delete_vector;
    }

    pub(crate) fn insert_equality_delete(
        &self,
        delete_file_path: &str,
        eq_del: Receiver<(Predicate, Option<EqDeleteKeySet>)>,
    ) {
        let notify = Arc::new(Notify::new());
        {
            let mut state = self.state.write().unwrap();
            state.equality_deletes.insert(
                delete_file_path.to_string(),
                EqDelState::Loading(notify.clone()),
            );
        }

        let state = self.state.clone();
        let delete_file_path = delete_file_path.to_string();
        crate::runtime::spawn(async move {
            let (predicate, key_set) = eq_del.await.unwrap();
            {
                let mut state = state.write().unwrap();
                state
                    .equality_deletes
                    .insert(delete_file_path, EqDelState::Loaded(predicate, key_set));
            }
            notify.notify_waiters();
        });
    }
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
    pub fn deleted_row_positions(&self, task: &FileScanTask) -> Option<Arc<Mutex<DeleteVector>>> {
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
        let positional_mask: Option<BooleanArray> = match self.get_delete_vector(task) {
            Some(deletes) => {
                let deletes = deletes.lock().map_err(|_| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "positional delete vector mutex was poisoned",
                    )
                })?;
                if deletes.is_empty() {
                    None
                } else {
                    // Range-walk the delete window — byte-identical to the per-row `!contains` probe,
                    // O(D_window) instead of O(num_rows). See `positional_delete_keep_mask`.
                    Some(positional_delete_keep_mask(&deletes, row_base, num_rows))
                }
            }
            None => None,
        };

        // Equality-delete predicate → a keep-mask (true ⇒ survives). A NULL under three-valued logic
        // is NOT a survivor (matches the Parquet RowFilter), so coerce nulls to false.
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

/// Coerce a three-valued keep-mask to two-valued: every NULL becomes `false` (drop the row), matching
/// the Parquet `RowFilter` (which never keeps a null result). Mirrors the reader's
/// `coerce_nulls_to_false`.
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
/// [`DeleteVectorIterator::advance_to`] has four sharp edges this routine must respect:
///
/// 1. It is a **no-op until the iterator has been primed** with at least one `next()` — it returns
///    early while `inner` is `None`. So we call `next()` once *before* any `advance_to`.
/// 2. It repositions the *underlying* iterator but cannot un-yield a value already pulled into our
///    local `cached`. So a primed `cached` that is already a **legitimate in-window** position
///    (`>= base`) must NOT be dropped — `advance_to` cannot rewind, so discarding it would lose a
///    real delete.
/// 3. `advance_to(target)` is only safe to drive **forward**: its inner loop is
///    `while inner.high_bits < hi`, so if the prime already pulled a value from a *higher* high-bits
///    group than `target`'s (the 2^32-straddle case — a window whose first delete sits above the
///    boundary while `base` sits below it), the loop is skipped and the call instead corrupts the
///    inner bitmap-iterator position via `bitmap_iter.advance_to(lo)`, silently dropping later
///    in-group deletes. (This latent edge does not bite the Parquet `build_deletes_row_selection`
///    path, which only ever advances to *monotonically increasing* targets.)
/// 4. `advance_to(base)` is a **hint, not a guarantee** of landing in-window: when *no* delete
///    reaches `base`'s high-bits group (every remaining delete is below the window), it walks
///    `outer` to exhaustion and returns, leaving the iterator on a still-below-window value. So the
///    post-advance `next()` may still yield `pos < base`.
///
/// We therefore call `advance_to(base)` **only when the primed `cached` is below the window**
/// (`cached < base`) — exactly the case where skipping is needed and `advance_to` is driven forward
/// (edge 3) — and refresh `cached` afterward. When `cached` is already `>= base` (or `None`), the
/// iterator is already positioned and we leave it untouched. Because of edge 4, the walk loop then
/// re-checks each `pos` against `base` and only flips when `base <= pos < end`, silently skipping any
/// residual below-window delete `advance_to` could not get past. This is a strict superset of
/// `build_deletes_row_selection`'s stale-cache refresh, hardened to be correct regardless of how far
/// `advance_to` actually managed to skip.
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
        assert_eq!(result.lock().unwrap().len(), 12); // pos dels from pos del file 1 and 2

        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_tasks[1].deletes, file_scan_tasks[1].schema_ref())
            .await
            .unwrap()
            .unwrap();

        let result = delete_filter
            .get_delete_vector(&file_scan_tasks[1])
            .unwrap();
        assert_eq!(result.lock().unwrap().len(), 8); // no pos dels for file 3
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
                data_file_path: format!("{}/1.parquet", table_location.to_str().unwrap()),
                data_file_format: DataFileFormat::Parquet,
                schema: data_file_schema.clone(),
                project_field_ids: vec![],
                predicate: None,
                deletes: vec![pos_del_1, pos_del_2.clone()],
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
                data_file_path: format!("{}/2.parquet", table_location.to_str().unwrap()),
                data_file_format: DataFileFormat::Parquet,
                schema: data_file_schema.clone(),
                project_field_ids: vec![],
                predicate: None,
                deletes: vec![pos_del_3],
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
            data_file_path: "data.parquet".to_string(),
            data_file_format: crate::spec::DataFileFormat::Parquet,
            schema: schema.clone(),
            project_field_ids: vec![],
            predicate: None,
            deletes: vec![FileScanTaskDeleteFile {
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
            }],
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
        filter.insert_equality_delete("eq-del.parquet", rx);

        // No key set (predicate-only path) for this case-sensitivity test.
        tx.send((pred, None)).unwrap();

        // ---------- should FAIL ----------
        let result = filter.build_equality_delete_predicate(&task).await;

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
        assert_eq!(positions.lock().unwrap().len(), 12);
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

    /// THE KEY-NULL BAIL FOR THE NEW TYPES: a Time / Fixed batch with a NULL in the key column makes
    /// the fast path return `None`, and the predicate fallback — which previously ERRORED for
    /// Time/Fixed — now SUCCEEDS (the new `get_arrow_datum` arms) and deletes the NULL row via 3VL +
    /// null-coercion even without a matching NULL delete tuple. This pins the (b)-leg of the gate
    /// admission: re-admitting Time/Fixed is sound only because the bail target no longer errors.
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
        // The predicate oracle for that batch must SUCCEED (no FeatureUnsupported) and delete the
        // NULL row: survival(NULL) = (NULL != t) = NULL → coerced false → deleted.
        let oracle = multi_col_oracle_deleted_mask(&["t"], schema, &delete_rows, &batch);
        assert_eq!(
            oracle,
            vec![true, false, true],
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
            vec![true, false, true],
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
    /// return `None` (route this batch to the predicate fallback). The predicate path deletes such a
    /// NULL row via 3VL + null-coercion EVEN WITHOUT a matching NULL delete tuple — which set
    /// membership would NOT reproduce — so the fallback is mandatory. This pins that exact contract.
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

        // And the predicate oracle for that same batch deletes the NULL row (3VL + null-coercion):
        // survival(NULL) = (NULL != 3) = NULL → coerced false → deleted.
        let oracle = multi_col_oracle_deleted_mask(&["v"], schema, &delete_rows, &batch);
        assert_eq!(
            oracle,
            vec![true, false, true],
            "predicate path deletes the NULL key-column row even without a NULL delete tuple — the \
             reason the set path must defer"
        );
    }

    /// THE GATE: Float / Double key columns must NOT build a set (route to the predicate fallback),
    /// and Decimal / Unknown are likewise excluded. This is what keeps the proven-divergent float
    /// case on the untouched predicate path. (Time and Fixed are NOT excluded — they gained a
    /// `get_arrow_datum` arm and their equality is integer-/byte-identical; see the eligible-type
    /// assertions below and `test_h6_set_time_matches_oracle` / `test_h6_set_fixed_matches_oracle`.)
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
}
