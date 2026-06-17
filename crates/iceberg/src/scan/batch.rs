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

//! The typed batch scan ([`BatchScan`]) — the Java `BatchScan` / `BatchScanAdapter` surface.
//!
//! In Java 1.10.0 `Table.newBatchScan()` returns a `BatchScanAdapter` that simply WRAPS the
//! `TableScan` from `newScan()`: every method delegates 1:1 to the underlying scan
//! (`BatchScanAdapter` bytecode — `planFiles`/`planTasks`/`useSnapshot`/`useRef`/`asOfTime`/...
//! all `getfield scan; invoke...`). So `BatchScan` is a thin TYPED adapter over the SAME
//! planning pipeline, NOT a separate planning engine:
//!
//! ```text
//! table.newBatchScan().planTasks()  ==  table.newScan().planTasks()
//! ```
//!
//! [`BatchScan`] mirrors that shape on top of this crate's [`TableScanBuilder`] /
//! [`TableScan`](super::TableScan): it carries a [`TableScanBuilder`] (the inherited
//! filter / select / case-sensitive / split-config knobs reuse it verbatim) and adds the
//! three BatchScan-specific time-travel selectors — [`use_snapshot`](BatchScan::use_snapshot),
//! [`use_ref`](BatchScan::use_ref), and [`as_of_time`](BatchScan::as_of_time). At plan time it
//! [`build`](TableScanBuilder::build)s the underlying [`TableScan`](super::TableScan) and
//! DELEGATES to its [`plan_files`](super::TableScan::plan_files) /
//! [`plan_tasks`](super::TableScan::plan_tasks) — the split + bin-pack logic is REUSED, never
//! reimplemented here.
//!
//! ## Selector semantics (Java `SnapshotScan.useSnapshot` / `useRef` / `asOfTime`, 1.10.0
//! bytecode-verified)
//!
//! The three selectors are mutually exclusive with FIRST-WINS / explicit-conflict semantics:
//! each one `checkArgument`s that no snapshot id has ALREADY been pinned and throws otherwise.
//! Concretely, in Java:
//!
//! - `useSnapshot(id)` — `checkArgument(snapshotId == null, "Cannot override snapshot, already
//!   set snapshot id=%s")`, then verifies the id exists ("Cannot find snapshot with ID %s").
//! - `useRef(name)` — `name.equals("main")` returns the table default (pins NOTHING); otherwise
//!   `checkArgument(snapshotId == null, "Cannot override ref, already set snapshot id=%s")`,
//!   then resolves the ref ("Cannot find ref %s").
//! - `asOfTime(ms)` — `checkArgument(snapshotId == null, ...)`, resolves
//!   `SnapshotUtil.snapshotIdAsOfTime(table, ms)` (the GREATEST `timestamp_ms <= ms` over the
//!   snapshot LOG; throws "Cannot find a snapshot older than %s" when none), then delegates to
//!   `useSnapshot(resolvedId)`.
//!
//! Java throws synchronously from the selector. Rust's builder methods are infallible
//! (`self -> Self`) so a conflicting selector cannot return an error in place; instead the
//! FIRST conflict is LATCHED and surfaced as a typed [`Error`] from
//! [`plan_files`](BatchScan::plan_files) / [`plan_tasks`](BatchScan::plan_tasks) (and any
//! invalid pin — unknown ref, unknown id, no-snapshot-as-of-time — surfaces from the same
//! point via the underlying [`build`](TableScanBuilder::build), exactly as Java surfaces the
//! id/ref existence checks). This preserves Java's behavior (a contradictory selector is a hard
//! error) while keeping the fluent builder shape.

use super::{FileScanTaskStream, TableScanBuilder};
use crate::scan::CombinedScanTaskStream;
use crate::spec::SnapshotRef;
use crate::table::Table;
use crate::{Error, ErrorKind, Result};

/// A typed batch scan over a [`Table`], mirroring Java `BatchScan` / `BatchScanAdapter`.
///
/// Build one with [`Table::batch_scan`]. It exposes the BatchScan-specific time-travel
/// selectors ([`use_snapshot`](Self::use_snapshot) / [`use_ref`](Self::use_ref) /
/// [`as_of_time`](Self::as_of_time)) and re-exposes the inherited scan knobs
/// ([`select`](Self::select) / [`with_filter`](Self::with_filter) /
/// [`with_case_sensitive`](Self::with_case_sensitive) / the split-config setters) by forwarding
/// to the wrapped [`TableScanBuilder`]. [`plan_files`](Self::plan_files) /
/// [`plan_tasks`](Self::plan_tasks) DELEGATE to the underlying
/// [`TableScan`](super::TableScan) pipeline (no reimplemented split/bin-pack).
pub struct BatchScan<'a> {
    /// The wrapped table-scan builder — the SAME builder `Table::scan()` returns. The inherited
    /// knobs (filter / select / case-sensitive / split config / concurrency / metrics) reuse it
    /// directly; only the time-travel selectors are layered on top.
    builder: TableScanBuilder<'a>,
    /// The selector that has already pinned a snapshot, if any — the FIRST-WINS anchor for the
    /// Java conflict checks. `None` ⇒ nothing pinned yet (the table default / current snapshot,
    /// matching `useRef("main")`).
    pinned_by: Option<Selector>,
    /// The FIRST conflict encountered while applying selectors, latched verbatim so it can be
    /// surfaced as a typed error at plan time (Java throws synchronously from the selector; the
    /// fluent Rust builder cannot, so it defers the first one). Once set it is never overwritten —
    /// the FIRST contradictory selector is the one Java would have thrown on.
    conflict: Option<Error>,
}

/// Which time-travel selector pinned the scan's snapshot — used only to render the Java
/// `checkArgument` conflict message ("Cannot override snapshot/ref, already set snapshot id=%s").
#[derive(Debug, Clone, Copy)]
enum Selector {
    /// [`BatchScan::use_snapshot`] pinned an explicit snapshot id (rendered in the conflict
    /// message, mirroring Java's `already set snapshot id=%s`).
    Snapshot(i64),
    /// [`BatchScan::use_ref`] pinned a (non-`main`) branch/tag.
    Ref,
    /// [`BatchScan::as_of_time`] pinned the snapshot current as of a timestamp.
    AsOfTime,
}

impl<'a> BatchScan<'a> {
    /// Creates a batch scan over `table` (Java `Table.newBatchScan()` → `new
    /// BatchScanAdapter(table.newScan())`). Starts from the same default-snapshot baseline as
    /// [`Table::scan`].
    pub(crate) fn new(table: &'a Table) -> Self {
        Self {
            builder: TableScanBuilder::new(table),
            pinned_by: None,
            conflict: None,
        }
    }

    // -- Time-travel selectors (Java `BatchScan.useSnapshot` / `useRef` / `asOfTime`). --

    /// Scan a specific snapshot by id (Java `BatchScan.useSnapshot(long)`).
    ///
    /// First-wins: if a snapshot has already been pinned by an earlier selector, this LATCHES a
    /// typed conflict (Java `checkArgument(snapshotId == null, "Cannot override snapshot, already
    /// set snapshot id=%s")`) surfaced at plan time. The id's existence is verified by the
    /// underlying [`build`](TableScanBuilder::build), like Java's "Cannot find snapshot with ID".
    pub fn use_snapshot(mut self, snapshot_id: i64) -> Self {
        if let Some(error) = self.conflict_if_pinned("snapshot") {
            self.conflict.get_or_insert(error);
            return self;
        }
        self.pinned_by = Some(Selector::Snapshot(snapshot_id));
        self.builder = self.builder.snapshot_id(snapshot_id);
        self
    }

    /// Scan the snapshot a branch or tag reference points to (Java `BatchScan.useRef(String)`).
    ///
    /// `"main"` resolves to the table default and pins NOTHING (matching Java's `useRef(MAIN)`
    /// early-return). Any other ref pins the scan first-wins: a prior pin LATCHES a typed conflict
    /// (Java "Cannot override ref, already set snapshot id=%s"); an unknown ref name surfaces from
    /// the underlying [`build`](TableScanBuilder::build) (Java "Cannot find ref %s").
    pub fn use_ref(mut self, ref_name: impl Into<String>) -> Self {
        let ref_name = ref_name.into();
        // Java: `useRef("main")` returns the table default WITHOUT pinning a snapshot id, so it is
        // NOT a conflict even after another selector ran. Mirror that no-op.
        if ref_name == crate::spec::MAIN_BRANCH {
            return self;
        }
        if let Some(error) = self.conflict_if_pinned("ref") {
            self.conflict.get_or_insert(error);
            return self;
        }
        self.pinned_by = Some(Selector::Ref);
        self.builder = self.builder.use_ref(ref_name);
        self
    }

    /// Scan the snapshot that was CURRENT as of `timestamp_ms` (Java `BatchScan.asOfTime(long)`).
    ///
    /// Resolves `SnapshotUtil.snapshotIdAsOfTime(table, timestamp_ms)`: the snapshot with the
    /// GREATEST `timestamp_ms` that is `<=` the argument, walked over the table's snapshot LOG
    /// (history) — the snapshot that was current at that wall-clock time. The boundary is
    /// INCLUSIVE (`<=`): a timestamp landing exactly on a log entry selects THAT entry.
    ///
    /// First-wins: a prior pin LATCHES a typed conflict (Java `checkArgument(snapshotId == null,
    /// "Cannot override snapshot, already set snapshot id=%s")`). When NO log entry is `<=`
    /// `timestamp_ms`, a typed [`Error`] is latched (Java "Cannot find a snapshot older than %s").
    pub fn as_of_time(mut self, timestamp_ms: i64) -> Self {
        if let Some(error) = self.conflict_if_pinned("snapshot") {
            self.conflict.get_or_insert(error);
            return self;
        }
        // Resolve over the snapshot LOG: keep the LAST entry whose timestamp is <= the argument.
        // The log is chronological, so the last such entry has the GREATEST timestamp <= ms — the
        // snapshot that was current at that time (Java `SnapshotUtil.nullableSnapshotIdAsOfTime`:
        // iterate `table.history()`, keep when `timestampMillis <= ms`, last assignment wins).
        // `rfind` walks the log from the back, so the FIRST match it returns is the greatest
        // timestamp <= ms — exactly the last-assignment-wins entry of Java's forward loop.
        let resolved = self
            .builder
            .table()
            .metadata()
            .history()
            .iter()
            .rfind(|entry| entry.timestamp_ms <= timestamp_ms)
            .map(|entry| entry.snapshot_id);

        match resolved {
            Some(snapshot_id) => {
                // Java delegates to `useSnapshot(resolvedId)` — pin it the same way.
                self.pinned_by = Some(Selector::AsOfTime);
                self.builder = self.builder.snapshot_id(snapshot_id);
            }
            None => {
                self.conflict.get_or_insert_with(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Cannot find a snapshot older than {timestamp_ms}"),
                    )
                });
            }
        }
        self
    }

    // -- Inherited scan knobs (Java `Scan`) — forwarded to the wrapped builder. --

    /// Select some columns of the table (Java `Scan.select`). Forwards to
    /// [`TableScanBuilder::select`].
    pub fn select(mut self, column_names: impl IntoIterator<Item = impl ToString>) -> Self {
        self.builder = self.builder.select(column_names);
        self
    }

    /// Select all columns (Java `Scan` default). Forwards to [`TableScanBuilder::select_all`].
    pub fn select_all(mut self) -> Self {
        self.builder = self.builder.select_all();
        self
    }

    /// Select no columns. Forwards to [`TableScanBuilder::select_empty`].
    pub fn select_empty(mut self) -> Self {
        self.builder = self.builder.select_empty();
        self
    }

    /// Apply a row filter (Java `Scan.filter`). Forwards to [`TableScanBuilder::with_filter`].
    pub fn with_filter(mut self, predicate: crate::expr::Predicate) -> Self {
        self.builder = self.builder.with_filter(predicate);
        self
    }

    /// Set the scan's case sensitivity (Java `Scan.caseSensitive`). Forwards to
    /// [`TableScanBuilder::with_case_sensitive`].
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.builder = self.builder.with_case_sensitive(case_sensitive);
        self
    }

    /// Override the split target size in bytes (Java `Scan.option(SPLIT_SIZE, ...)`). Forwards to
    /// [`TableScanBuilder::with_split_size`].
    pub fn with_split_size(mut self, split_size: u64) -> Self {
        self.builder = self.builder.with_split_size(split_size);
        self
    }

    /// Override the split planning lookback (Java `Scan.option(SPLIT_LOOKBACK, ...)`). Forwards to
    /// [`TableScanBuilder::with_split_lookback`].
    pub fn with_split_lookback(mut self, split_lookback: usize) -> Self {
        self.builder = self.builder.with_split_lookback(split_lookback);
        self
    }

    /// Override the per-open file cost (Java `Scan.option(SPLIT_OPEN_FILE_COST, ...)`). Forwards
    /// to [`TableScanBuilder::with_split_open_file_cost`].
    pub fn with_split_open_file_cost(mut self, split_open_file_cost: u64) -> Self {
        self.builder = self.builder.with_split_open_file_cost(split_open_file_cost);
        self
    }

    // -- Delegated planning (Java `BatchScanAdapter.planFiles` / `planTasks` → `scan.*`). --

    /// Returns the scan's [`FileScanTask`](super::FileScanTask) stream, DELEGATING to the
    /// underlying [`TableScan::plan_files`](super::TableScan::plan_files) (Java
    /// `BatchScanAdapter.planFiles()` → `scan.planFiles()`).
    pub async fn plan_files(self) -> Result<FileScanTaskStream> {
        self.into_table_scan()?.plan_files().await
    }

    /// Returns the scan's [`CombinedScanTask`](super::CombinedScanTask) GROUP stream (split +
    /// bin-packed), DELEGATING to the underlying
    /// [`TableScan::plan_tasks`](super::TableScan::plan_tasks) (Java
    /// `BatchScanAdapter.planTasks()` → `scan.planTasks()`). The split/bin-pack logic is REUSED
    /// from the [`TableScan`](super::TableScan) pipeline, not reimplemented here.
    pub async fn plan_tasks(self) -> Result<CombinedScanTaskStream> {
        self.into_table_scan()?.plan_tasks().await
    }

    /// Returns the snapshot this batch scan will read (Java `BatchScan.snapshot()` →
    /// `scan.snapshot()`). `None` for a snapshotless table. A latched selector conflict surfaces
    /// here as a typed error.
    pub fn snapshot(self) -> Result<Option<SnapshotRef>> {
        Ok(self.into_table_scan()?.snapshot().cloned())
    }

    // -- internals --

    /// Builds the underlying [`TableScan`](super::TableScan), surfacing any latched selector
    /// conflict FIRST (Java throws the `checkArgument` synchronously, before planning).
    fn into_table_scan(self) -> Result<super::TableScan> {
        if let Some(error) = self.conflict {
            return Err(error);
        }
        self.builder.build()
    }

    /// Returns the Java `checkArgument` conflict error for applying `selector_kind` after a
    /// snapshot is already pinned, or `None` when nothing is pinned yet. The message mirrors
    /// Java's two strings: `useRef` says "Cannot override ref, ..."; `useSnapshot`/`asOfTime`
    /// say "Cannot override snapshot, ...".
    fn conflict_if_pinned(&self, selector_kind: &str) -> Option<Error> {
        let pinned = self.pinned_by?;
        let already = match pinned {
            Selector::Snapshot(id) => format!("snapshot id={id}"),
            Selector::AsOfTime => "a snapshot resolved from a timestamp".to_string(),
            Selector::Ref => "a snapshot resolved from a ref".to_string(),
        };
        Some(Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot override {selector_kind}, already set {already}"),
        ))
    }
}
