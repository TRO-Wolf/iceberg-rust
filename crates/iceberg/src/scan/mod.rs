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

//! Table scan api.

mod batch;
mod bin_pack;
mod cache;
use cache::*;
mod context;
use context::*;
mod incremental;
pub use batch::*;
pub use incremental::*;
mod metrics_collector;
mod task;
mod task_group;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::RecordBatch;
use futures::channel::mpsc::{Sender, channel};
use futures::stream::BoxStream;
use futures::{SinkExt, StreamExt, TryStreamExt};
pub use task::*;
pub use task_group::*;

use crate::arrow::ArrowReaderBuilder;
use crate::delete_file_index::DeleteFileIndex;
use crate::events::{self, ScanEvent};
use crate::expr::visitors::inclusive_metrics_evaluator::InclusiveMetricsEvaluator;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::io::FileIO;
use crate::metadata_columns::{get_metadata_field_id, is_metadata_column_name};
use crate::metrics::{MetricsReport, MetricsReporter, ScanReport, TimeUnit, TimerResult};
use crate::runtime::spawn;
use crate::scan::metrics_collector::ScanMetricsCollector;
use crate::spec::{DataContentType, ManifestContentType, SnapshotRef};
use crate::table::Table;
use crate::utils::available_parallelism;
use crate::{Error, ErrorKind, Result};

/// A stream of arrow [`RecordBatch`]es.
pub type ArrowRecordBatchStream = BoxStream<'static, Result<RecordBatch>>;

/// A stream of [`CombinedScanTask`]s — the output of [`TableScan::plan_tasks`].
pub type CombinedScanTaskStream = BoxStream<'static, Result<CombinedScanTask>>;

/// Java `TableProperties.SPLIT_SIZE` — the per-bin split target in bytes.
pub(crate) const PROPERTY_SPLIT_SIZE: &str = "read.split.target-size";
/// Java `TableProperties.SPLIT_SIZE_DEFAULT` — 128 MiB.
pub(crate) const PROPERTY_SPLIT_SIZE_DEFAULT: u64 = 134_217_728;
/// Java `TableProperties.SPLIT_LOOKBACK` — the max number of simultaneously-open bins.
pub(crate) const PROPERTY_SPLIT_LOOKBACK: &str = "read.split.planning-lookback";
/// Java `TableProperties.SPLIT_LOOKBACK_DEFAULT` — 10.
pub(crate) const PROPERTY_SPLIT_LOOKBACK_DEFAULT: usize = 10;
/// Java `TableProperties.SPLIT_OPEN_FILE_COST` — the per-open cost in bytes used in the weight floor.
pub(crate) const PROPERTY_SPLIT_OPEN_FILE_COST: &str = "read.split.open-file-cost";
/// Java `TableProperties.SPLIT_OPEN_FILE_COST_DEFAULT` — 4 MiB.
pub(crate) const PROPERTY_SPLIT_OPEN_FILE_COST_DEFAULT: u64 = 4_194_304;

/// Builder to create table scan.
pub struct TableScanBuilder<'a> {
    table: &'a Table,
    // Defaults to none which means select all columns
    column_names: Option<Vec<String>>,
    snapshot_id: Option<i64>,
    snapshot_ref: Option<String>,
    batch_size: Option<usize>,
    case_sensitive: bool,
    filter: Option<Predicate>,
    concurrency_limit_data_files: usize,
    concurrency_limit_manifest_entries: usize,
    concurrency_limit_manifest_files: usize,
    row_group_filtering_enabled: bool,
    row_selection_enabled: bool,
    metrics_reporter: Option<Arc<dyn MetricsReporter>>,
    /// Scan-time OVERRIDE for the split target size (Java `TableScanContext.option(SPLIT_SIZE)`).
    /// `None` ⇒ fall back to the table property then the Java default at [`plan_tasks`] time.
    split_size: Option<u64>,
    /// Scan-time OVERRIDE for the split planning lookback (Java
    /// `TableScanContext.option(SPLIT_LOOKBACK)`).
    split_lookback: Option<usize>,
    /// Scan-time OVERRIDE for the split open-file cost (Java
    /// `TableScanContext.option(SPLIT_OPEN_FILE_COST)`).
    split_open_file_cost: Option<u64>,
}

impl<'a> TableScanBuilder<'a> {
    pub(crate) fn new(table: &'a Table) -> Self {
        let num_cpus = available_parallelism().get();

        Self {
            table,
            column_names: None,
            snapshot_id: None,
            snapshot_ref: None,
            batch_size: None,
            case_sensitive: true,
            filter: None,
            concurrency_limit_data_files: num_cpus,
            concurrency_limit_manifest_entries: num_cpus,
            concurrency_limit_manifest_files: num_cpus,
            row_group_filtering_enabled: true,
            row_selection_enabled: false,
            metrics_reporter: None,
            split_size: None,
            split_lookback: None,
            split_open_file_cost: None,
        }
    }

    /// Overrides the split target size (bytes) used by [`TableScan::plan_tasks`], mirroring Java's
    /// scan-time `option(TableProperties.SPLIT_SIZE, ...)`.
    ///
    /// When unset, [`plan_tasks`](TableScan::plan_tasks) reads the table property
    /// `read.split.target-size`, defaulting to 128 MiB (Java `SPLIT_SIZE_DEFAULT`). The override
    /// wins over the table property, which wins over the default — exactly Java's
    /// `BaseScan.targetSplitSize()`.
    pub fn with_split_size(mut self, split_size: u64) -> Self {
        self.split_size = Some(split_size);
        self
    }

    /// Overrides the split planning lookback (the max number of simultaneously-open bins) used by
    /// [`TableScan::plan_tasks`], mirroring Java's `option(TableProperties.SPLIT_LOOKBACK, ...)`.
    ///
    /// When unset, [`plan_tasks`](TableScan::plan_tasks) reads `read.split.planning-lookback`,
    /// defaulting to 10 (Java `SPLIT_LOOKBACK_DEFAULT`).
    pub fn with_split_lookback(mut self, split_lookback: usize) -> Self {
        self.split_lookback = Some(split_lookback);
        self
    }

    /// Overrides the per-open file cost (bytes) used in the bin-packing weight floor by
    /// [`TableScan::plan_tasks`], mirroring Java's `option(TableProperties.SPLIT_OPEN_FILE_COST, ...)`.
    ///
    /// When unset, [`plan_tasks`](TableScan::plan_tasks) reads `read.split.open-file-cost`,
    /// defaulting to 4 MiB (Java `SPLIT_OPEN_FILE_COST_DEFAULT`).
    pub fn with_split_open_file_cost(mut self, split_open_file_cost: u64) -> Self {
        self.split_open_file_cost = Some(split_open_file_cost);
        self
    }

    /// Sets the desired size of batches in the response
    /// to something other than the default
    pub fn with_batch_size(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the scan's case sensitivity
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Specifies a predicate to use as a filter
    pub fn with_filter(mut self, predicate: Predicate) -> Self {
        // calls rewrite_not to remove Not nodes, which must be absent
        // when applying the manifest evaluator
        self.filter = Some(predicate.rewrite_not());
        self
    }

    /// Select all columns.
    pub fn select_all(mut self) -> Self {
        self.column_names = None;
        self
    }

    /// Select empty columns.
    pub fn select_empty(mut self) -> Self {
        self.column_names = Some(vec![]);
        self
    }

    /// Select some columns of the table.
    pub fn select(mut self, column_names: impl IntoIterator<Item = impl ToString>) -> Self {
        self.column_names = Some(
            column_names
                .into_iter()
                .map(|item| item.to_string())
                .collect(),
        );
        self
    }

    /// Set the snapshot to scan. When not set, it uses current snapshot.
    pub fn snapshot_id(mut self, snapshot_id: i64) -> Self {
        self.snapshot_id = Some(snapshot_id);
        self
    }

    /// Returns the [`Table`] this builder scans. Used by the [`BatchScan`](crate::scan::BatchScan)
    /// adapter to resolve `as_of_time` over the table's snapshot log before pinning the snapshot
    /// id via [`snapshot_id`](Self::snapshot_id).
    pub(crate) fn table(&self) -> &Table {
        self.table
    }

    /// Scan the snapshot that a branch or tag reference points to.
    ///
    /// Mirrors Java `TableScan.useRef(String ref)`: the named reference is
    /// resolved to its snapshot at [`build`](Self::build) time. The `"main"`
    /// branch resolves to the table's current snapshot, matching Java's
    /// `useRef(MAIN_BRANCH)` returning the table default.
    ///
    /// Combining `use_ref` with [`snapshot_id`](Self::snapshot_id) is rejected
    /// at `build` time — a reference and an explicit snapshot id are
    /// contradictory selectors (Java "Cannot override ref, already set snapshot
    /// id"). An unknown reference name is also rejected at `build` time.
    ///
    /// # Arguments
    ///
    /// * `ref_name` - the branch or tag name to resolve.
    pub fn use_ref(mut self, ref_name: impl Into<String>) -> Self {
        self.snapshot_ref = Some(ref_name.into());
        self
    }

    /// Sets the concurrency limit for both manifest files and manifest
    /// entries for this scan
    pub fn with_concurrency_limit(mut self, limit: usize) -> Self {
        self.concurrency_limit_manifest_files = limit;
        self.concurrency_limit_manifest_entries = limit;
        self.concurrency_limit_data_files = limit;
        self
    }

    /// Sets the data file concurrency limit for this scan
    pub fn with_data_file_concurrency_limit(mut self, limit: usize) -> Self {
        self.concurrency_limit_data_files = limit;
        self
    }

    /// Sets the manifest entry concurrency limit for this scan
    pub fn with_manifest_entry_concurrency_limit(mut self, limit: usize) -> Self {
        self.concurrency_limit_manifest_entries = limit;
        self
    }

    /// Determines whether to enable row group filtering.
    /// When enabled, if a read is performed with a filter predicate,
    /// then the metadata for each row group in the parquet file is
    /// evaluated against the filter predicate and row groups
    /// that cant contain matching rows will be skipped entirely.
    ///
    /// Defaults to enabled, as it generally improves performance or
    /// keeps it the same, with performance degradation unlikely.
    pub fn with_row_group_filtering_enabled(mut self, row_group_filtering_enabled: bool) -> Self {
        self.row_group_filtering_enabled = row_group_filtering_enabled;
        self
    }

    /// Determines whether to enable row selection.
    /// When enabled, if a read is performed with a filter predicate,
    /// then (for row groups that have not been skipped) the page index
    /// for each row group in a parquet file is parsed and evaluated
    /// against the filter predicate to determine if ranges of rows
    /// within a row group can be skipped, based upon the page-level
    /// statistics for each column.
    ///
    /// Defaults to being disabled. Enabling requires parsing the parquet page
    /// index, which can be slow enough that parsing the page index outweighs any
    /// gains from the reduced number of rows that need scanning.
    /// It is recommended to experiment with partitioning, sorting, row group size,
    /// page size, and page row limit Iceberg settings on the table being scanned in
    /// order to get the best performance from using row selection.
    pub fn with_row_selection_enabled(mut self, row_selection_enabled: bool) -> Self {
        self.row_selection_enabled = row_selection_enabled;
        self
    }

    /// Configures a [`MetricsReporter`] to receive a [`ScanReport`] once the scan's
    /// [`FileScanTask`] stream has been fully consumed.
    ///
    /// Mirrors Java `TableScanContext.metricsReporter` (set via `Catalog`/`TableScan`
    /// configuration). When set, [`plan_files`](TableScan::plan_files) collects the
    /// scan-planning metrics — manifest-list totals, scanned/skipped manifests, produced
    /// data/delete file counts and sizes, and the total planning duration — and reports a
    /// single [`MetricsReport::Scan`] when the returned stream completes (the analogue of
    /// Java's `CloseableIterable.whenComplete(...)` close hook).
    ///
    /// **Opt-in.** When this is not called, planning installs no collector, no timer, and
    /// no stream wrapper: the scan path is byte-for-byte the un-instrumented path (Java's
    /// `ScanMetrics.noop()` when no reporter is configured).
    pub fn with_metrics_reporter(mut self, reporter: Arc<dyn MetricsReporter>) -> Self {
        self.metrics_reporter = Some(reporter);
        self
    }

    /// Resolves the split planning config (target / lookback / open-file-cost) at `build` time,
    /// porting Java `BaseScan.targetSplitSize()` / `splitLookback()` / `splitOpenFileCost()`:
    /// read the table property (defaulting to the hard-coded Java default), then let the scan-time
    /// override win over the SAME key. Validation (target > 0, lookback > 0, open-file-cost >= 0)
    /// is enforced at [`plan_tasks`](TableScan::plan_tasks) time, matching Java's
    /// `TableScanUtil.validatePlanningArguments`.
    fn resolve_split_config(&self) -> SplitConfig {
        let props = self.table.metadata().properties();

        let split_size = self.split_size.unwrap_or_else(|| {
            props
                .get(PROPERTY_SPLIT_SIZE)
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(PROPERTY_SPLIT_SIZE_DEFAULT)
        });
        let split_lookback = self.split_lookback.unwrap_or_else(|| {
            props
                .get(PROPERTY_SPLIT_LOOKBACK)
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(PROPERTY_SPLIT_LOOKBACK_DEFAULT)
        });
        let split_open_file_cost = self.split_open_file_cost.unwrap_or_else(|| {
            props
                .get(PROPERTY_SPLIT_OPEN_FILE_COST)
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(PROPERTY_SPLIT_OPEN_FILE_COST_DEFAULT)
        });

        SplitConfig {
            split_size,
            split_lookback,
            split_open_file_cost,
        }
    }

    /// Build the table scan.
    pub fn build(self) -> Result<TableScan> {
        let split_config = self.resolve_split_config();

        // Capture the table name once, for the `ScanEvent` fired at plan time (Java
        // `SnapshotScan` reads `table().name()`). Captured here so both the snapshotless and
        // the normal `TableScan` carry it.
        let table_name = self.table.identifier().to_string();

        // Resolve a branch/tag reference (`use_ref`) to a concrete snapshot id,
        // honoring BOTH selectors (Java `SnapshotScan.useRef` L116-128):
        //   - a ref AND an explicit snapshot id are contradictory selectors
        //     (Java "Cannot override ref, already set snapshot id");
        //   - an unknown ref name is rejected (Java "Cannot find ref %s").
        // `snapshot_for_ref("main")` resolves to the current snapshot, matching
        // Java's `useRef(MAIN_BRANCH)` returning the table default.
        let snapshot_id = match (self.snapshot_id, self.snapshot_ref.as_ref()) {
            (Some(_), Some(ref_name)) => {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot scan using both a ref ('{ref_name}') and a snapshot id at the same time"
                    ),
                ));
            }
            (None, Some(ref_name)) => {
                let snapshot = self
                    .table
                    .metadata()
                    .snapshot_for_ref(ref_name)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("snapshot ref '{ref_name}' not found"),
                        )
                    })?;
                Some(snapshot.snapshot_id())
            }
            (snapshot_id, None) => snapshot_id,
        };

        let snapshot = match snapshot_id {
            Some(snapshot_id) => self
                .table
                .metadata()
                .snapshot_by_id(snapshot_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Snapshot with id {snapshot_id} not found"),
                    )
                })?
                .clone(),
            None => {
                let Some(current_snapshot_id) = self.table.metadata().current_snapshot() else {
                    // No snapshot ⇒ an empty scan with no rows. Java
                    // `SnapshotScan.planFiles` returns `CloseableIterable.empty()` BEFORE
                    // the timer/report block, so NO `ScanReport` is emitted for a
                    // snapshotless table — match that by leaving `metrics: None`.
                    return Ok(TableScan {
                        batch_size: self.batch_size,
                        column_names: self.column_names,
                        file_io: self.table.file_io().clone(),
                        plan_context: None,
                        table_name,
                        concurrency_limit_data_files: self.concurrency_limit_data_files,
                        concurrency_limit_manifest_entries: self.concurrency_limit_manifest_entries,
                        concurrency_limit_manifest_files: self.concurrency_limit_manifest_files,
                        row_group_filtering_enabled: self.row_group_filtering_enabled,
                        row_selection_enabled: self.row_selection_enabled,
                        metrics: None,
                        split_config,
                    });
                };
                current_snapshot_id.clone()
            }
        };

        let schema = snapshot.schema(self.table.metadata())?;

        // Check that all column names exist in the schema (skip reserved columns).
        if let Some(column_names) = self.column_names.as_ref() {
            for column_name in column_names {
                // Skip reserved columns that don't exist in the schema
                if is_metadata_column_name(column_name) {
                    continue;
                }
                if schema.field_by_name(column_name).is_none() {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Column {column_name} not found in table. Schema: {schema}"),
                    ));
                }
            }
        }

        let mut field_ids = vec![];
        let column_names = self.column_names.clone().unwrap_or_else(|| {
            schema
                .as_struct()
                .fields()
                .iter()
                .map(|f| f.name.clone())
                .collect()
        });

        for column_name in column_names.iter() {
            // Handle metadata columns (like "_file")
            if is_metadata_column_name(column_name) {
                field_ids.push(get_metadata_field_id(column_name)?);
                continue;
            }

            let field_id = schema.field_id_by_name(column_name).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Column {column_name} not found in table. Schema: {schema}"),
                )
            })?;

            schema
                .as_struct()
                .field_by_id(field_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::FeatureUnsupported,
                        format!(
                        "Column {column_name} is not a direct child of schema but a nested field, which is not supported now. Schema: {schema}"
                    ),
                )
            })?;

            field_ids.push(field_id);
        }

        let snapshot_bound_predicate = if let Some(ref predicates) = self.filter {
            Some(predicates.bind(schema.clone(), true)?)
        } else {
            None
        };

        // Build the opt-in metrics context (collector + report inputs) ONLY when a
        // reporter was configured. When `None`, no collector is threaded into planning, so
        // the scan path is byte-for-byte unchanged. The report inputs mirror exactly what
        // Java `SnapshotScan.planFiles` reads to build its `ImmutableScanReport`:
        // schema id, projected field ids + their column names, table name, snapshot id,
        // and the filter (defaulting to `alwaysTrue` when the scan has no filter, like
        // Java `BaseScan.filter()`).
        let field_ids = Arc::new(field_ids);
        let metrics = self.metrics_reporter.clone().map(|reporter| {
            let projected_field_names = field_ids
                .iter()
                .map(|field_id| {
                    schema
                        .name_by_field_id(*field_id)
                        .unwrap_or_default()
                        .to_string()
                })
                .collect();
            ScanMetricsContext {
                reporter,
                collector: Arc::new(ScanMetricsCollector::new()),
                table_name: self.table.identifier().to_string(),
                snapshot_id: snapshot.snapshot_id(),
                schema_id: schema.schema_id(),
                projected_field_ids: field_ids.to_vec(),
                projected_field_names,
                filter: self.filter.clone().unwrap_or(Predicate::AlwaysTrue),
            }
        });

        // Parse the table's default name mapping (property `schema.name-mapping.default`) ONCE per
        // plan (Java `NameMappingParser.fromJson`); the shared `Arc` is threaded onto every
        // produced `FileScanTask` so the arrow reader can resolve field ids by column name for
        // id-less data files. An unparsable value is a loud typed error naming the property.
        let name_mapping = parse_name_mapping(self.table.metadata())?;

        let plan_context = PlanContext {
            snapshot,
            table_metadata: self.table.metadata_ref(),
            snapshot_schema: schema,
            case_sensitive: self.case_sensitive,
            predicate: self.filter.map(Arc::new),
            snapshot_bound_predicate: snapshot_bound_predicate.map(Arc::new),
            object_cache: self.table.object_cache(),
            field_ids,
            name_mapping,
            partition_filter_cache: Arc::new(PartitionFilterCache::new()),
            manifest_evaluator_cache: Arc::new(ManifestEvaluatorCache::new()),
            expression_evaluator_cache: Arc::new(ExpressionEvaluatorCache::new()),
            metrics_collector: metrics.as_ref().map(|context| context.collector.clone()),
        };

        Ok(TableScan {
            batch_size: self.batch_size,
            column_names: self.column_names,
            file_io: self.table.file_io().clone(),
            plan_context: Some(plan_context),
            table_name,
            concurrency_limit_data_files: self.concurrency_limit_data_files,
            concurrency_limit_manifest_entries: self.concurrency_limit_manifest_entries,
            concurrency_limit_manifest_files: self.concurrency_limit_manifest_files,
            row_group_filtering_enabled: self.row_group_filtering_enabled,
            row_selection_enabled: self.row_selection_enabled,
            metrics,
            split_config,
        })
    }
}

/// The resolved split planning config — Java `BaseScan.targetSplitSize` / `splitLookback` /
/// `splitOpenFileCost`. Captured once at [`build`](TableScanBuilder::build) time so
/// [`TableScan::plan_tasks`] is a pure function of the captured config.
#[derive(Debug, Clone, Copy)]
struct SplitConfig {
    split_size: u64,
    split_lookback: usize,
    split_open_file_cost: u64,
}

/// Table scan.
#[derive(Debug)]
pub struct TableScan {
    /// A [PlanContext], if this table has at least one snapshot, otherwise None.
    ///
    /// If this is None, then the scan contains no rows.
    plan_context: Option<PlanContext>,
    /// The fully-qualified table name (`identifier().to_string()`), captured at
    /// [`build`](TableScanBuilder::build) time. Threaded in solely so [`plan_files`] can
    /// construct the [`ScanEvent`] without re-deriving it (Java `SnapshotScan` reads
    /// `table().name()`). `TableScan` no longer holds the `Table` itself.
    table_name: String,
    batch_size: Option<usize>,
    file_io: FileIO,
    column_names: Option<Vec<String>>,
    /// The maximum number of manifest files that will be
    /// retrieved from [`FileIO`] concurrently
    concurrency_limit_manifest_files: usize,

    /// The maximum number of [`ManifestEntry`]s that will
    /// be processed in parallel
    concurrency_limit_manifest_entries: usize,

    /// The maximum number of [`ManifestEntry`]s that will
    /// be processed in parallel
    concurrency_limit_data_files: usize,

    row_group_filtering_enabled: bool,
    row_selection_enabled: bool,

    /// Everything needed to collect and emit a [`ScanReport`], present only when the scan
    /// opted in via [`TableScanBuilder::with_metrics_reporter`] AND the table has a
    /// snapshot to scan. `None` ⇒ no metrics are collected and `plan_files` is unchanged.
    metrics: Option<ScanMetricsContext>,

    /// The resolved split planning config (target / lookback / open-file-cost) used by
    /// [`Self::plan_tasks`]. Captured at [`build`](TableScanBuilder::build) time; never consulted by
    /// [`Self::plan_files`] (whose path is byte-unchanged).
    split_config: SplitConfig,
}

/// The reporter, the shared counter collector, and the immutable inputs needed to build a
/// [`ScanReport`] at stream-completion time.
///
/// Bundled so the opt-in is a single `Option` on [`TableScan`]: when it is `None`,
/// [`TableScan::plan_files`] installs no collector, no timer, and no stream wrapper, and
/// the planning path is byte-for-byte the un-instrumented path. The report metadata
/// (table name / snapshot id / schema id / projected ids+names / filter) is captured once
/// at [`build`](TableScanBuilder::build) time — exactly the values Java's
/// `SnapshotScan.planFiles` reads to build its `ImmutableScanReport`.
#[derive(Clone)]
struct ScanMetricsContext {
    reporter: Arc<dyn MetricsReporter>,
    collector: Arc<ScanMetricsCollector>,
    table_name: String,
    snapshot_id: i64,
    schema_id: i32,
    projected_field_ids: Vec<i32>,
    projected_field_names: Vec<String>,
    filter: Predicate,
}

// `dyn MetricsReporter` is not `Debug`, so derive cannot apply to `ScanMetricsContext`
// (and `TableScan` derives `Debug`). Render the report inputs and elide the reporter +
// the live collector.
impl std::fmt::Debug for ScanMetricsContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanMetricsContext")
            .field("table_name", &self.table_name)
            .field("snapshot_id", &self.snapshot_id)
            .field("schema_id", &self.schema_id)
            .field("projected_field_ids", &self.projected_field_ids)
            .field("projected_field_names", &self.projected_field_names)
            .field("filter", &self.filter)
            .finish_non_exhaustive()
    }
}

impl TableScan {
    /// Returns a stream of [`FileScanTask`]s.
    pub async fn plan_files(&self) -> Result<FileScanTaskStream> {
        let Some(plan_context) = self.plan_context.as_ref() else {
            return Ok(Box::pin(futures::stream::empty()));
        };

        // Fire the `ScanEvent` (Java `SnapshotScan.planFiles` fires it when `snapshot != null`,
        // before `doPlanFiles`). Placed AFTER the snapshotless guard above, so an empty scan
        // fires nothing. The filter is the UNBOUND row filter (Java fires `context.rowFilter()`,
        // the unbound expression — NOT the snapshot-bound one), defaulting to `AlwaysTrue` when
        // the scan has no filter. Like Java, the call is unguarded: a panicking listener
        // propagates out of `plan_files`.
        events::notify_all(&ScanEvent::new(
            self.table_name.clone(),
            plan_context.snapshot.snapshot_id(),
            plan_context
                .predicate
                .as_ref()
                .map(|p| p.as_ref().clone())
                .unwrap_or(Predicate::AlwaysTrue),
            plan_context.snapshot_schema.clone(),
        ));

        // Start the planning timer when (and only when) metrics are enabled — Java
        // `SnapshotScan.planFiles` starts `scanMetrics().totalPlanningDuration()` before
        // `doPlanFiles()`. When `self.metrics` is `None` this stays `None` and nothing
        // below is instrumented (the opt-in / byte-unchanged path).
        let planning_started_at = self.metrics.as_ref().map(|_| Instant::now());

        let concurrency_limit_manifest_files = self.concurrency_limit_manifest_files;
        let concurrency_limit_manifest_entries = self.concurrency_limit_manifest_entries;

        // used to stream ManifestEntryContexts between stages of the file plan operation
        let (manifest_entry_data_ctx_tx, manifest_entry_data_ctx_rx) =
            channel(concurrency_limit_manifest_files);
        let (manifest_entry_delete_ctx_tx, manifest_entry_delete_ctx_rx) =
            channel(concurrency_limit_manifest_files);

        // used to stream the results back to the caller
        let (file_scan_task_tx, file_scan_task_rx) = channel(concurrency_limit_manifest_entries);

        let (delete_file_idx, delete_file_tx) = DeleteFileIndex::new();

        let manifest_list = plan_context.get_manifest_list().await?;

        // Count the manifest-list entries by content type — Java
        // `DataTableScan.doPlanFiles` increments `totalDataManifests`/`totalDeleteManifests`
        // by `dataManifests.size()`/`deleteManifests.size()` of the snapshot's list. A
        // no-op when no collector is configured.
        if let Some(metrics) = self.metrics.as_ref() {
            let (data_manifests, delete_manifests) =
                manifest_list
                    .entries()
                    .iter()
                    .fold((0i64, 0i64), |(data, deletes), entry| match entry.content {
                        ManifestContentType::Data => (data + 1, deletes),
                        ManifestContentType::Deletes => (data, deletes + 1),
                    });
            metrics
                .collector
                .add_total_manifests(data_manifests, delete_manifests);
        }

        // get the [`ManifestFile`]s from the [`ManifestList`], filtering out any
        // whose partitions cannot match this
        // scan's filter
        let manifest_file_contexts = plan_context.build_manifest_file_contexts(
            manifest_list,
            manifest_entry_data_ctx_tx,
            delete_file_idx.clone(),
            manifest_entry_delete_ctx_tx,
        )?;

        let mut channel_for_manifest_error = file_scan_task_tx.clone();

        // Concurrently load all [`Manifest`]s and stream their [`ManifestEntry`]s
        spawn(async move {
            let result = futures::stream::iter(manifest_file_contexts)
                .try_for_each_concurrent(concurrency_limit_manifest_files, |ctx| async move {
                    ctx.fetch_manifest_and_stream_manifest_entries().await
                })
                .await;

            if let Err(error) = result {
                let _ = channel_for_manifest_error.send(Err(error)).await;
            }
        });

        let mut channel_for_data_manifest_entry_error = file_scan_task_tx.clone();
        let mut channel_for_delete_manifest_entry_error = file_scan_task_tx.clone();

        // Process the delete file [`ManifestEntry`] stream in parallel
        spawn(async move {
            let result = manifest_entry_delete_ctx_rx
                .map(|me_ctx| Ok((me_ctx, delete_file_tx.clone())))
                .try_for_each_concurrent(
                    concurrency_limit_manifest_entries,
                    |(manifest_entry_context, tx)| async move {
                        spawn(async move {
                            Self::process_delete_manifest_entry(manifest_entry_context, tx).await
                        })
                        .await
                    },
                )
                .await;

            if let Err(error) = result {
                let _ = channel_for_delete_manifest_entry_error
                    .send(Err(error))
                    .await;
            }
        })
        .await;

        // Process the data file [`ManifestEntry`] stream in parallel
        spawn(async move {
            let result = manifest_entry_data_ctx_rx
                .map(|me_ctx| Ok((me_ctx, file_scan_task_tx.clone())))
                .try_for_each_concurrent(
                    concurrency_limit_manifest_entries,
                    |(manifest_entry_context, tx)| async move {
                        spawn(async move {
                            Self::process_data_manifest_entry(manifest_entry_context, tx).await
                        })
                        .await
                    },
                )
                .await;

            if let Err(error) = result {
                let _ = channel_for_data_manifest_entry_error.send(Err(error)).await;
            }
        });

        // When metrics are enabled, wrap the result stream so each produced task is counted
        // as it is yielded (Java `ScanMetricsUtil.fileTask` in the lazy `createFileScanTasks`
        // transform) and a single `ScanReport` is emitted when the stream is fully consumed
        // (Java's `CloseableIterable.whenComplete` close hook). When `self.metrics` is
        // `None`, return the inner stream verbatim — byte-for-byte the un-instrumented path.
        match (self.metrics.clone(), planning_started_at) {
            (Some(metrics), Some(planning_started_at)) => {
                Ok(Box::pin(MetricsReportingFileScanTaskStream::new(
                    file_scan_task_rx,
                    metrics,
                    planning_started_at,
                )))
            }
            _ => Ok(file_scan_task_rx.boxed()),
        }
    }

    /// Plans the scan into [`CombinedScanTask`] GROUPS — split + bin-packed — porting Java
    /// `Scan.planTasks()` (1.10.0).
    ///
    /// This is the split-and-group layer that sits ABOVE [`plan_files`](Self::plan_files): it
    /// drives `plan_files()` UNCHANGED (no reporter side effects, the byte-for-byte path), splits
    /// each produced [`FileScanTask`] at the split target ([`FileScanTask::split`], Java
    /// `TableScanUtil.splitFiles`), then bin-packs the resulting sub-tasks into groups with the
    /// `largestBinFirst` packer (Java `TableScanUtil.planTasks` →
    /// `BinPacking.PackingIterable(..., largestBinFirst = true)`).
    ///
    /// The arguments come from the resolved [`SplitConfig`] captured at build time (table property
    /// then scan-time override then the Java default — see
    /// [`with_split_size`](TableScanBuilder::with_split_size) etc.). They are validated exactly as
    /// Java `TableScanUtil.validatePlanningArguments`: `target > 0`, `lookback > 0`,
    /// `open_file_cost >= 0` (always true for a `u64`). The weight of each task is
    /// [`FileScanTask::weight`] — `max(length + deleteBytes, (1 + #deletes) * openFileCost)`.
    ///
    /// The returned stream yields one [`CombinedScanTask`] per emitted bin, in the bin-packer's
    /// emission order. This is the plain data-file group path: each [`CombinedScanTask`] carries NO
    /// grouping key and the tasks are NOT merged (Java's partition-aware
    /// `planTaskGroups(groupingKeyType)` overload is out of scope).
    ///
    /// Because the packer is order-sensitive (FIFO bin selection + largest-bin eviction), the
    /// FileScanTask stream is COLLECTED in arrival order before packing — matching Java, which
    /// drives the packer over the in-order `splitFiles(planFiles())` iterable.
    pub async fn plan_tasks(&self) -> Result<CombinedScanTaskStream> {
        let SplitConfig {
            split_size,
            split_lookback,
            split_open_file_cost,
        } = self.split_config;

        // Java `TableScanUtil.validatePlanningArguments`: target > 0, lookback > 0,
        // open-file-cost >= 0 (a `u64` is always >= 0). A misconfiguration is a hard error.
        if split_size == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Split size must be > 0: {split_size}"),
            ));
        }
        if split_lookback == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Split planning lookback must be > 0: {split_lookback}"),
            ));
        }

        // Drive plan_files() UNCHANGED and collect its tasks in arrival order (the packer is
        // order-sensitive; Java drives the bin-packer over the in-order splitFiles(planFiles())).
        let file_scan_tasks: Vec<FileScanTask> = self.plan_files().await?.try_collect().await?;

        // splitFiles: expand each whole-file task into its target-sized sub-tasks, preserving
        // arrival order (Java `FluentIterable.transformAndConcat`).
        let mut split_tasks: Vec<FileScanTask> = Vec::with_capacity(file_scan_tasks.len());
        for task in file_scan_tasks {
            split_tasks.extend(task.split(split_size)?);
        }

        // Bin-pack with largestBinFirst = true (the planTasks path). The weight is the
        // FileScanTask weight under this scan's open-file cost.
        let groups: Vec<CombinedScanTask> = bin_pack::PackingIterator::new(
            split_tasks.into_iter(),
            split_size,
            split_lookback,
            true,
            move |task: &FileScanTask| task.weight(split_open_file_cost),
        )
        .map(CombinedScanTask::new)
        .collect();

        Ok(Box::pin(futures::stream::iter(groups.into_iter().map(Ok))))
    }

    /// Returns an [`ArrowRecordBatchStream`].
    pub async fn to_arrow(&self) -> Result<ArrowRecordBatchStream> {
        let mut arrow_reader_builder = ArrowReaderBuilder::new(self.file_io.clone())
            .with_data_file_concurrency_limit(self.concurrency_limit_data_files)
            .with_row_group_filtering_enabled(self.row_group_filtering_enabled)
            .with_row_selection_enabled(self.row_selection_enabled);

        if let Some(batch_size) = self.batch_size {
            arrow_reader_builder = arrow_reader_builder.with_batch_size(batch_size);
        }

        arrow_reader_builder.build().read(self.plan_files().await?)
    }

    /// Returns a reference to the column names of the table scan.
    pub fn column_names(&self) -> Option<&[String]> {
        self.column_names.as_deref()
    }

    /// Returns a reference to the snapshot of the table scan.
    pub fn snapshot(&self) -> Option<&SnapshotRef> {
        self.plan_context.as_ref().map(|x| &x.snapshot)
    }

    async fn process_data_manifest_entry(
        manifest_entry_context: ManifestEntryContext,
        mut file_scan_task_tx: Sender<Result<FileScanTask>>,
    ) -> Result<()> {
        // skip processing this manifest entry if it has been marked as deleted
        if !manifest_entry_context.manifest_entry.is_alive() {
            return Ok(());
        }

        // abort the plan if we encounter a manifest entry for a delete file
        if manifest_entry_context.manifest_entry.content_type() != DataContentType::Data {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Encountered an entry for a delete file in a data file manifest",
            ));
        }

        if let Some(ref bound_predicates) = manifest_entry_context.bound_predicates {
            let BoundPredicates {
                snapshot_bound_predicate,
                partition_bound_predicate,
            } = bound_predicates.as_ref();

            let expression_evaluator_cache =
                manifest_entry_context.expression_evaluator_cache.as_ref();

            let expression_evaluator = expression_evaluator_cache.get(
                manifest_entry_context.partition_spec_id,
                partition_bound_predicate,
            )?;

            // skip any data file whose partition data indicates that it can't contain
            // any data that matches this scan's filter
            if !expression_evaluator.eval(manifest_entry_context.manifest_entry.data_file())? {
                return Ok(());
            }

            // skip any data file whose metrics don't match this scan's filter
            if !InclusiveMetricsEvaluator::eval(
                snapshot_bound_predicate,
                manifest_entry_context.manifest_entry.data_file(),
                false,
            )? {
                return Ok(());
            }
        }

        // congratulations! the manifest entry has made its way through the
        // entire plan without getting filtered out. Create a corresponding
        // FileScanTask and push it to the result stream
        file_scan_task_tx
            .send(Ok(manifest_entry_context.into_file_scan_task().await?))
            .await?;

        Ok(())
    }

    async fn process_delete_manifest_entry(
        manifest_entry_context: ManifestEntryContext,
        mut delete_file_ctx_tx: Sender<DeleteFileContext>,
    ) -> Result<()> {
        // skip processing this manifest entry if it has been marked as deleted
        if !manifest_entry_context.manifest_entry.is_alive() {
            return Ok(());
        }

        // abort the plan if we encounter a manifest entry that is not for a delete file
        if manifest_entry_context.manifest_entry.content_type() == DataContentType::Data {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Encountered an entry for a data file in a delete manifest",
            ));
        }

        if let Some(ref bound_predicates) = manifest_entry_context.bound_predicates {
            let expression_evaluator_cache =
                manifest_entry_context.expression_evaluator_cache.as_ref();

            let expression_evaluator = expression_evaluator_cache.get(
                manifest_entry_context.partition_spec_id,
                &bound_predicates.partition_bound_predicate,
            )?;

            // skip any data file whose partition data indicates that it can't contain
            // any data that matches this scan's filter
            if !expression_evaluator.eval(manifest_entry_context.manifest_entry.data_file())? {
                return Ok(());
            }
        }

        delete_file_ctx_tx
            .send(DeleteFileContext {
                manifest_entry: manifest_entry_context.manifest_entry.clone(),
                partition_spec_id: manifest_entry_context.partition_spec_id,
            })
            .await?;

        Ok(())
    }
}

impl ScanMetricsContext {
    /// Records a single produced [`FileScanTask`] in the collector, mirroring Java
    /// `ScanMetricsUtil.fileTask`: bump `result_data_files`, accumulate the data file's
    /// size, count the task's attached delete files, and accumulate their sizes.
    fn record_task(&self, task: &FileScanTask) {
        let delete_file_size_in_bytes: u64 = task
            .deletes
            .iter()
            .map(|delete| delete.file_size_in_bytes)
            .sum();
        self.collector.record_file_task(
            task.file_size_in_bytes as i64,
            task.deletes.len() as i64,
            delete_file_size_in_bytes as i64,
        );
    }

    /// Snapshots the collector, stamps the planning duration, builds the [`ScanReport`],
    /// and reports it to the configured [`MetricsReporter`] exactly once.
    ///
    /// Mirrors the close hook of Java `SnapshotScan.planFiles`: stop the planning timer,
    /// build an `ImmutableScanReport` from the captured table name / snapshot id / schema
    /// id / projected ids+names / filter and the collected `ScanMetricsResult`, then call
    /// `metricsReporter().report(scanReport)`.
    fn emit_report(&self, planning_started_at: Instant) {
        let elapsed = planning_started_at.elapsed();
        let mut scan_metrics = self.collector.snapshot();
        // The timer is reported in nanoseconds with a single timed event, matching Java's
        // `totalPlanningDuration` (a `Timer` declared with `TimeUnit.NANOSECONDS`, one
        // `start()`/`stop()` per scan).
        scan_metrics.total_planning_duration =
            Some(TimerResult::new(TimeUnit::Nanoseconds, elapsed, 1));

        let report = ScanReport {
            table_name: self.table_name.clone(),
            snapshot_id: self.snapshot_id,
            filter: self.filter.clone(),
            schema_id: self.schema_id,
            projected_field_ids: self.projected_field_ids.clone(),
            projected_field_names: self.projected_field_names.clone(),
            scan_metrics,
            metadata: HashMap::new(),
        };

        self.reporter.report(MetricsReport::Scan(report));
    }
}

/// A [`FileScanTaskStream`] that records each yielded task in its collector and reports a
/// single [`ScanReport`] when the inner stream is fully consumed.
///
/// This is the Rust analogue of Java's `CloseableIterable.whenComplete(doPlanFiles(), ...)`
/// in `SnapshotScan.planFiles`: per-task accounting happens lazily as each task is pulled
/// (Java's `createFileScanTasks` transform), and the report is emitted exactly ONCE, on the
/// transition to stream exhaustion (Java's close hook). It is only ever constructed when a
/// reporter was configured, so it adds no overhead to the un-instrumented scan path.
struct MetricsReportingFileScanTaskStream {
    inner: futures::channel::mpsc::Receiver<Result<FileScanTask>>,
    metrics: ScanMetricsContext,
    planning_started_at: Instant,
    /// `true` once the report has been emitted, so exhaustion (or a re-poll after `None`)
    /// reports at most once.
    reported: bool,
}

impl MetricsReportingFileScanTaskStream {
    fn new(
        inner: futures::channel::mpsc::Receiver<Result<FileScanTask>>,
        metrics: ScanMetricsContext,
        planning_started_at: Instant,
    ) -> Self {
        Self {
            inner,
            metrics,
            planning_started_at,
            reported: false,
        }
    }
}

impl futures::Stream for MetricsReportingFileScanTaskStream {
    type Item = Result<FileScanTask>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;

        match self.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(task))) => {
                self.metrics.record_task(&task);
                Poll::Ready(Some(Ok(task)))
            }
            Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(error))),
            Poll::Ready(None) => {
                // Stream fully consumed — emit the single scan report (once).
                if !self.reported {
                    self.reported = true;
                    self.metrics.emit_report(self.planning_started_at);
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

pub(crate) struct BoundPredicates {
    partition_bound_predicate: BoundPredicate,
    snapshot_bound_predicate: BoundPredicate,
}

#[cfg(test)]
pub mod tests {
    //! shared tests for the table scan API
    #![allow(missing_docs)]

    use std::collections::{BTreeSet, HashMap};
    use std::fs;
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{
        Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch,
        StringArray,
    };
    use futures::{TryStreamExt, stream};
    use minijinja::value::Value;
    use minijinja::{AutoEscape, Environment, context};
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;
    use uuid::Uuid;

    use crate::TableIdent;
    use crate::arrow::ArrowReaderBuilder;
    use crate::expr::{Bind, BoundPredicate, Predicate, Reference};
    use crate::io::{FileIO, OutputFile};
    use crate::metadata_columns::RESERVED_COL_NAME_FILE;
    use crate::metrics::{InMemoryMetricsReporter, MetricUnit, MetricsReport};
    use crate::scan::{CombinedScanTask, FileScanTask, FileScanTaskStream};
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Datum, Literal, ManifestEntry,
        ManifestListWriter, ManifestStatus, ManifestWriterBuilder, NestedField, PartitionSpec,
        PrimitiveType, Schema, SnapshotReference, SnapshotRetention, Struct, StructType,
        TableMetadata, TableProperties, Transform, Type, UnboundPartitionField,
    };
    use crate::table::Table;

    fn render_template(template: &str, ctx: Value) -> String {
        let mut env = Environment::new();
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.render_str(template, ctx).unwrap()
    }

    /// Decodes a scan-output column to a flat [`Int64Array`], transparently
    /// expanding the run-end-encoded constant array the reader produces for an
    /// identity-partitioned column (Java `PartitionUtil.constantsMap`). A plain
    /// `Int64Array` passes through unchanged. Used by the read tests so they assert
    /// on logical values regardless of whether a column is a materialized partition
    /// constant or read from the data file. `pub` because the incremental-scan
    /// read pins (`incremental.rs`) reuse it.
    pub fn decode_int64_column(column: &ArrayRef) -> Int64Array {
        arrow_cast::cast::cast(column, &arrow_schema::DataType::Int64)
            .expect("column is castable to Int64")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("cast yields an Int64Array")
            .clone()
    }

    pub struct TableTestFixture {
        pub table_location: String,
        pub table: Table,
    }

    impl TableTestFixture {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            let tmp_dir = TempDir::new().unwrap();
            let table_location = tmp_dir.path().join("table1");
            let manifest_list1_location = table_location.join("metadata/manifests_list_1.avro");
            let manifest_list2_location = table_location.join("metadata/manifests_list_2.avro");
            let table_metadata1_location = table_location.join("metadata/v1.json");

            let file_io = FileIO::new_with_fs();

            let table_metadata = {
                let template_json_str = fs::read_to_string(format!(
                    "{}/testdata/example_table_metadata_v2.json",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap();
                let metadata_json = render_template(&template_json_str, context! {
                    table_location => &table_location,
                    manifest_list_1_location => &manifest_list1_location,
                    manifest_list_2_location => &manifest_list2_location,
                    table_metadata_1_location => &table_metadata1_location,
                });
                serde_json::from_str::<TableMetadata>(&metadata_json).unwrap()
            };

            let table = Table::builder()
                .metadata(table_metadata)
                .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
                .file_io(file_io.clone())
                .metadata_location(table_metadata1_location.as_os_str().to_str().unwrap())
                .build()
                .unwrap();

            Self {
                table_location: table_location.to_str().unwrap().to_string(),
                table,
            }
        }

        #[allow(clippy::new_without_default)]
        pub fn new_empty() -> Self {
            let tmp_dir = TempDir::new().unwrap();
            let table_location = tmp_dir.path().join("table1");
            let table_metadata1_location = table_location.join("metadata/v1.json");

            let file_io = FileIO::new_with_fs();

            let table_metadata = {
                let template_json_str = fs::read_to_string(format!(
                    "{}/testdata/example_empty_table_metadata_v2.json",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap();
                let metadata_json = render_template(&template_json_str, context! {
                    table_location => &table_location,
                    table_metadata_1_location => &table_metadata1_location,
                });
                serde_json::from_str::<TableMetadata>(&metadata_json).unwrap()
            };

            let table = Table::builder()
                .metadata(table_metadata)
                .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
                .file_io(file_io.clone())
                .metadata_location(table_metadata1_location.as_os_str().to_str().unwrap())
                .build()
                .unwrap();

            Self {
                table_location: table_location.to_str().unwrap().to_string(),
                table,
            }
        }

        pub fn new_unpartitioned() -> Self {
            let tmp_dir = TempDir::new().unwrap();
            let table_location = tmp_dir.path().join("table1");
            let manifest_list1_location = table_location.join("metadata/manifests_list_1.avro");
            let manifest_list2_location = table_location.join("metadata/manifests_list_2.avro");
            let table_metadata1_location = table_location.join("metadata/v1.json");

            let file_io = FileIO::new_with_fs();

            let mut table_metadata = {
                let template_json_str = fs::read_to_string(format!(
                    "{}/testdata/example_table_metadata_v2.json",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap();
                let metadata_json = render_template(&template_json_str, context! {
                    table_location => &table_location,
                    manifest_list_1_location => &manifest_list1_location,
                    manifest_list_2_location => &manifest_list2_location,
                    table_metadata_1_location => &table_metadata1_location,
                });
                serde_json::from_str::<TableMetadata>(&metadata_json).unwrap()
            };

            table_metadata.default_spec = Arc::new(PartitionSpec::unpartition_spec());
            table_metadata.partition_specs.clear();
            table_metadata.default_partition_type = StructType::new(vec![]);
            table_metadata
                .partition_specs
                .insert(0, table_metadata.default_spec.clone());

            let table = Table::builder()
                .metadata(table_metadata)
                .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
                .file_io(file_io.clone())
                .metadata_location(table_metadata1_location.to_str().unwrap())
                .build()
                .unwrap();

            Self {
                table_location: table_location.to_str().unwrap().to_string(),
                table,
            }
        }

        /// Builds an UNPARTITIONED V2 fixture (schema `x,y,z,a,dbl,i32,i64,bool`, ids 1..8) with the
        /// `schema.name-mapping.default` property set to `name_mapping_json`. Used by the name-mapping
        /// scan-wiring pins: an unpartitioned base is required so the reader does NOT materialize
        /// identity-partition columns as constants (which would bypass the file for `x`), and the
        /// property drives [`parse_name_mapping`] during `build()`. Pass a deliberately-invalid JSON
        /// to pin the loud-failure path.
        pub fn new_with_name_mapping_property(name_mapping_json: &str) -> Self {
            let tmp_dir = TempDir::new().unwrap();
            let table_location = tmp_dir.path().join("table1");
            let manifest_list1_location = table_location.join("metadata/manifests_list_1.avro");
            let manifest_list2_location = table_location.join("metadata/manifests_list_2.avro");
            let table_metadata1_location = table_location.join("metadata/v1.json");

            let file_io = FileIO::new_with_fs();

            let mut table_metadata = {
                let template_json_str = fs::read_to_string(format!(
                    "{}/testdata/example_table_metadata_v2.json",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap();
                let metadata_json = render_template(&template_json_str, context! {
                    table_location => &table_location,
                    manifest_list_1_location => &manifest_list1_location,
                    manifest_list_2_location => &manifest_list2_location,
                    table_metadata_1_location => &table_metadata1_location,
                });
                serde_json::from_str::<TableMetadata>(&metadata_json).unwrap()
            };

            // Unpartition (see the doc comment: keep `x` a file column, not a partition constant).
            table_metadata.default_spec = Arc::new(PartitionSpec::unpartition_spec());
            table_metadata.partition_specs.clear();
            table_metadata.default_partition_type = StructType::new(vec![]);
            table_metadata
                .partition_specs
                .insert(0, table_metadata.default_spec.clone());

            // Set the name-mapping property the scan planner parses.
            table_metadata.properties.insert(
                TableProperties::PROPERTY_DEFAULT_NAME_MAPPING.to_string(),
                name_mapping_json.to_string(),
            );

            let table = Table::builder()
                .metadata(table_metadata)
                .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
                .file_io(file_io.clone())
                .metadata_location(table_metadata1_location.to_str().unwrap())
                .build()
                .unwrap();

            Self {
                table_location: table_location.to_str().unwrap().to_string(),
                table,
            }
        }

        /// Builds a fixture partitioned by `truncate(x, 100)` over the same schema
        /// and data as [`Self::new`]. Used to prove the residual wiring on a
        /// NON-identity (transform) partition: the parquet `x` column is `[1; 1024]`,
        /// so every row lands in `truncate(1, 100) == 0`.
        pub fn new_truncate_partitioned() -> Self {
            let tmp_dir = TempDir::new().unwrap();
            let table_location = tmp_dir.path().join("table1");
            let manifest_list1_location = table_location.join("metadata/manifests_list_1.avro");
            let manifest_list2_location = table_location.join("metadata/manifests_list_2.avro");
            let table_metadata1_location = table_location.join("metadata/v1.json");

            let file_io = FileIO::new_with_fs();

            let mut table_metadata = {
                let template_json_str = fs::read_to_string(format!(
                    "{}/testdata/example_table_metadata_v2.json",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap();
                let metadata_json = render_template(&template_json_str, context! {
                    table_location => &table_location,
                    manifest_list_1_location => &manifest_list1_location,
                    manifest_list_2_location => &manifest_list2_location,
                    table_metadata_1_location => &table_metadata1_location,
                });
                serde_json::from_str::<TableMetadata>(&metadata_json).unwrap()
            };

            let truncate_spec = Arc::new(
                PartitionSpec::builder(table_metadata.current_schema().clone())
                    .with_spec_id(0)
                    .add_unbound_field(
                        UnboundPartitionField::builder()
                            .source_id(1)
                            .name("x_trunc".to_string())
                            .field_id(1000)
                            .transform(Transform::Truncate(100))
                            .build(),
                    )
                    .unwrap()
                    .build()
                    .unwrap(),
            );
            let partition_type = truncate_spec
                .partition_type(table_metadata.current_schema())
                .unwrap();
            table_metadata.default_spec = truncate_spec.clone();
            table_metadata.default_partition_type = partition_type;
            table_metadata.partition_specs.clear();
            table_metadata.partition_specs.insert(0, truncate_spec);

            let table = Table::builder()
                .metadata(table_metadata)
                .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
                .file_io(file_io.clone())
                .metadata_location(table_metadata1_location.to_str().unwrap())
                .build()
                .unwrap();

            Self {
                table_location: table_location.to_str().unwrap().to_string(),
                table,
            }
        }

        /// Writes a single live data file (1.parquet) in partition
        /// `truncate(x, 100) == 0`, matching the parquet `x` column (`[1; 1024]`).
        /// Companion to [`Self::new_truncate_partitioned`].
        pub async fn setup_truncate_manifest_files(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let current_partition_spec = self.table.metadata().default_partition_spec();

            let parquet_file_size = self.write_parquet_data_files();

            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                current_partition_spec.as_ref().clone(),
            )
            .build_v2_data();
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/1.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                // truncate(x=1, 100) == 0.
                                .partition(Struct::from_iter([Some(Literal::long(0))]))
                                .key_metadata(None)
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            let data_file_manifest = writer.write_manifest_file().await.unwrap();

            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![data_file_manifest].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }

        /// Builds a fixture whose table metadata has TWO partition specs: the
        /// template's `identity(x)` (spec id 0) and an UNPARTITIONED spec (id 1) set
        /// as the table DEFAULT. The live data file is written under the NON-default
        /// spec 0. This isolates the "file's own spec vs the table default spec"
        /// choice: the residual must be computed from spec 0 (the file's), not the
        /// default spec 1 — a manifest written under an older spec is exactly what
        /// partition evolution produces.
        pub fn new_with_evolved_default_spec() -> Self {
            let tmp_dir = TempDir::new().unwrap();
            let table_location = tmp_dir.path().join("table1");
            let manifest_list1_location = table_location.join("metadata/manifests_list_1.avro");
            let manifest_list2_location = table_location.join("metadata/manifests_list_2.avro");
            let table_metadata1_location = table_location.join("metadata/v1.json");

            let file_io = FileIO::new_with_fs();

            let mut table_metadata = {
                let template_json_str = fs::read_to_string(format!(
                    "{}/testdata/example_table_metadata_v2.json",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap();
                let metadata_json = render_template(&template_json_str, context! {
                    table_location => &table_location,
                    manifest_list_1_location => &manifest_list1_location,
                    manifest_list_2_location => &manifest_list2_location,
                    table_metadata_1_location => &table_metadata1_location,
                });
                serde_json::from_str::<TableMetadata>(&metadata_json).unwrap()
            };

            // Add an unpartitioned spec (id 1) and make it the DEFAULT, while keeping
            // the template's identity(x) spec (id 0) in the spec map. The live file is
            // written under spec 0 below.
            let unpartitioned_spec = Arc::new(
                PartitionSpec::builder(table_metadata.current_schema().clone())
                    .with_spec_id(1)
                    .build()
                    .unwrap(),
            );
            table_metadata.default_spec = unpartitioned_spec.clone();
            table_metadata.default_partition_type = StructType::new(vec![]);
            table_metadata.partition_specs.insert(1, unpartitioned_spec);

            let table = Table::builder()
                .metadata(table_metadata)
                .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
                .file_io(file_io.clone())
                .metadata_location(table_metadata1_location.to_str().unwrap())
                .build()
                .unwrap();

            Self {
                table_location: table_location.to_str().unwrap().to_string(),
                table,
            }
        }

        /// Builds a fixture whose metadata carries TWO identity specs over DIFFERENT
        /// source columns: the template's `identity(x)` (spec 0) and a new `identity(y)`
        /// (spec 1, set as the table default). Used by the multi-spec scan test together
        /// with [`Self::setup_two_identity_spec_manifests`]: each data file is written in
        /// its OWN manifest under its OWN spec, so the scan must materialize each file's
        /// identity-partition constant under that file's spec (Java `file.spec()` per
        /// `ManifestFile.partition_spec_id`), not under one shared/default spec.
        pub fn new_with_two_identity_specs() -> Self {
            let tmp_dir = TempDir::new().unwrap();
            let table_location = tmp_dir.path().join("table1");
            let manifest_list1_location = table_location.join("metadata/manifests_list_1.avro");
            let manifest_list2_location = table_location.join("metadata/manifests_list_2.avro");
            let table_metadata1_location = table_location.join("metadata/v1.json");

            let file_io = FileIO::new_with_fs();

            let mut table_metadata = {
                let template_json_str = fs::read_to_string(format!(
                    "{}/testdata/example_table_metadata_v2.json",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap();
                let metadata_json = render_template(&template_json_str, context! {
                    table_location => &table_location,
                    manifest_list_1_location => &manifest_list1_location,
                    manifest_list_2_location => &manifest_list2_location,
                    table_metadata_1_location => &table_metadata1_location,
                });
                serde_json::from_str::<TableMetadata>(&metadata_json).unwrap()
            };

            // Add spec 1 = identity(y) (source_id 2) and make it the default. Spec 0
            // (identity(x)) stays in the spec map for the file written under it.
            let identity_y_spec = Arc::new(
                PartitionSpec::builder(table_metadata.current_schema().clone())
                    .with_spec_id(1)
                    .add_unbound_field(
                        UnboundPartitionField::builder()
                            .source_id(2)
                            .name("y".to_string())
                            .field_id(1000)
                            .transform(Transform::Identity)
                            .build(),
                    )
                    .unwrap()
                    .build()
                    .unwrap(),
            );
            let identity_y_partition_type = identity_y_spec
                .partition_type(table_metadata.current_schema())
                .unwrap();
            table_metadata.default_spec = identity_y_spec.clone();
            table_metadata.default_partition_type = identity_y_partition_type;
            table_metadata.partition_specs.insert(1, identity_y_spec);

            let table = Table::builder()
                .metadata(table_metadata)
                .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
                .file_io(file_io.clone())
                .metadata_location(table_metadata1_location.to_str().unwrap())
                .build()
                .unwrap();

            Self {
                table_location: table_location.to_str().unwrap().to_string(),
                table,
            }
        }

        /// Writes two live data files for the multi-spec scan test, each in its OWN
        /// manifest under its OWN spec, with a partition value that DIFFERS from the
        /// parquet data so the constant source is observable:
        ///  * 1.parquet under `identity(x)` (spec 0), partition `x == 777` (file x is `1`).
        ///  * 2.parquet under `identity(y)` (spec 1), partition `y == 888` (file y starts `2`).
        ///
        /// A correct multi-spec scan returns x=777 for the first file's rows and y=888 for
        /// the second file's rows — each under that file's own spec.
        pub async fn setup_two_identity_spec_manifests(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let spec_zero = self
                .table
                .metadata()
                .partition_spec_by_id(0)
                .expect("spec 0 (identity(x)) must exist")
                .clone();
            let spec_one = self
                .table
                .metadata()
                .partition_spec_by_id(1)
                .expect("spec 1 (identity(y)) must exist")
                .clone();

            let parquet_file_size = self.write_parquet_data_files();

            // Manifest A: 1.parquet under spec 0 (identity(x)), partition x == 777.
            let mut writer_a = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                spec_zero.as_ref().clone(),
            )
            .build_v2_data();
            writer_a
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/1.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                .partition(Struct::from_iter([Some(Literal::long(777))]))
                                .key_metadata(None)
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            let manifest_a = writer_a.write_manifest_file().await.unwrap();

            // Manifest B: 2.parquet under spec 1 (identity(y)), partition y == 888.
            let mut writer_b = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                spec_one.as_ref().clone(),
            )
            .build_v2_data();
            writer_b
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(1)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/2.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                .partition(Struct::from_iter([Some(Literal::long(888))]))
                                .key_metadata(None)
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            let manifest_b = writer_b.write_manifest_file().await.unwrap();

            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![manifest_a, manifest_b].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }

        /// Writes a single live data file (1.parquet) under the NON-default
        /// `identity(x)` spec (id 0, partition `x == 1`), matching the parquet `x`
        /// column (`[1; 1024]`). Companion to [`Self::new_with_evolved_default_spec`].
        pub async fn setup_manifest_files_under_spec_zero(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            // Write the manifest under spec 0 (identity(x)), NOT the default spec 1.
            let spec_zero = self
                .table
                .metadata()
                .partition_spec_by_id(0)
                .expect("spec 0 must exist")
                .clone();

            let parquet_file_size = self.write_parquet_data_files();

            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                spec_zero.as_ref().clone(),
            )
            .build_v2_data();
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/1.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                // identity(x) == 1, matching the parquet `x` column.
                                .partition(Struct::from_iter([Some(Literal::long(1))]))
                                .key_metadata(None)
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            let data_file_manifest = writer.write_manifest_file().await.unwrap();

            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![data_file_manifest].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }

        /// Writes a single live data file (1.parquet) under `identity(x)` (spec 0) whose
        /// manifest partition value (`x == 999`) DELIBERATELY DIFFERS from the parquet `x`
        /// column (`[1; 1024]`). This makes the scan output observably distinguish the two
        /// sources: with the identity-partition constants map ACTIVE the scan returns the
        /// PARTITION value (999, Java `PartitionUtil.constantsMap`); a reader that read `x`
        /// from the file would return 1. The metadata value is authoritative for an
        /// identity-partition column.
        pub async fn setup_manifest_files_partition_value_differs(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let current_partition_spec = self.table.metadata().default_partition_spec();

            let parquet_file_size = self.write_parquet_data_files();

            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                current_partition_spec.as_ref().clone(),
            )
            .build_v2_data();
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/1.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                // Partition value 999 != the parquet `x` column ([1; 1024]).
                                .partition(Struct::from_iter([Some(Literal::long(999))]))
                                .key_metadata(None)
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            let data_file_manifest = writer.write_manifest_file().await.unwrap();

            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![data_file_manifest].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }

        /// Writes a single live data file (1.parquet) under `identity(x)` (spec 0) whose
        /// manifest partition value is NULL. With the constants map active, the
        /// identity-partition `x` column materializes as all-null (Iceberg column-projection
        /// rule #4 — a null partition value is not added to the constants map, so the column
        /// resolves to null), rather than erroring. Companion to
        /// [`Self::setup_manifest_files_partition_value_differs`].
        pub async fn setup_manifest_files_null_partition(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let current_partition_spec = self.table.metadata().default_partition_spec();

            let parquet_file_size = self.write_parquet_data_files();

            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                current_partition_spec.as_ref().clone(),
            )
            .build_v2_data();
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/1.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                // NULL partition value for the identity(x) column.
                                .partition(Struct::from_iter([None]))
                                .key_metadata(None)
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            let data_file_manifest = writer.write_manifest_file().await.unwrap();

            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![data_file_manifest].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }

        fn next_manifest_file(&self) -> OutputFile {
            self.table
                .file_io()
                .new_output(format!(
                    "{}/metadata/manifest_{}.avro",
                    self.table_location,
                    Uuid::new_v4()
                ))
                .unwrap()
        }

        pub async fn setup_manifest_files(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let parent_snapshot = current_snapshot
                .parent_snapshot(self.table.metadata())
                .unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let current_partition_spec = self.table.metadata().default_partition_spec();

            // Write the data files first, then use the file size in the manifest entries
            let parquet_file_size = self.write_parquet_data_files();

            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                current_partition_spec.as_ref().clone(),
            )
            .build_v2_data();
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/1.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                // The table is `identity(x)`-partitioned and the reader now
                                // materializes identity-partition columns from this metadata
                                // value (Java `PartitionUtil.constantsMap`). The parquet `x`
                                // column is `[1; 1024]`, so the partition value MUST be 1 to keep
                                // the fixture internally consistent.
                                .partition(Struct::from_iter([Some(Literal::long(1))]))
                                .key_metadata(None)
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            writer
                .add_delete_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Deleted)
                        .snapshot_id(parent_snapshot.snapshot_id())
                        .sequence_number(parent_snapshot.sequence_number())
                        .file_sequence_number(parent_snapshot.sequence_number())
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/2.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                // Consistent with the parquet `x` column (`[1; 1024]`); see the
                                // note on 1.parquet above.
                                .partition(Struct::from_iter([Some(Literal::long(1))]))
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            writer
                .add_existing_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Existing)
                        .snapshot_id(parent_snapshot.snapshot_id())
                        .sequence_number(parent_snapshot.sequence_number())
                        .file_sequence_number(parent_snapshot.sequence_number())
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/3.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                // Consistent with the parquet `x` column (`[1; 1024]`); see the
                                // note on 1.parquet above.
                                .partition(Struct::from_iter([Some(Literal::long(1))]))
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            let data_file_manifest = writer.write_manifest_file().await.unwrap();

            // Write to manifest list
            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![data_file_manifest].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }

        /// Builds a planning fixture for `plan_tasks` (split + bin-pack): writes ONE real parquet
        /// payload to `1.parquet`, then declares FOUR live data file entries pointing at it whose
        /// manifest-entry `file_size_in_bytes` (and, for one, `split_offsets`) are HAND-SET so the
        /// split + bin-pack behavior is deterministic and independent of the real parquet byte size.
        ///
        /// The planning path reads only `file_size_in_bytes` (→ `length`), `data_file_format`,
        /// `split_offsets`, and the delete attachments — never the file bytes — so a single shared
        /// payload with overridden declared sizes is a faithful planning fixture. The four entries:
        ///   * `big` 1000 bytes, split offsets `[0, 400, 800]`  (offsets-aware split target)
        ///   * `mid` 600 bytes,  no offsets                     (fixed-size split on a small target)
        ///   * `small_a` 100 bytes, `small_b` 100 bytes         (pack together)
        ///
        /// All four are partition `x == 1` (consistent with the real `x = [1;1024]` payload).
        pub async fn setup_manifest_for_planning(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let current_partition_spec = self.table.metadata().default_partition_spec();

            // Write the single real parquet payload all four entries point at.
            self.write_parquet_data_files();

            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                current_partition_spec.as_ref().clone(),
            )
            .build_v2_data();

            let entry = |size: u64, offsets: Option<Vec<i64>>| {
                let mut builder = DataFileBuilder::default();
                builder
                    .partition_spec_id(0)
                    .content(DataContentType::Data)
                    .file_path(format!("{}/1.parquet", &self.table_location))
                    .file_format(DataFileFormat::Parquet)
                    .file_size_in_bytes(size)
                    .record_count(1)
                    .partition(Struct::from_iter([Some(Literal::long(1))]))
                    .key_metadata(None);
                if offsets.is_some() {
                    builder.split_offsets(offsets);
                }
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(builder.build().unwrap())
                    .build()
            };

            writer
                .add_entry(entry(1000, Some(vec![0, 400, 800])))
                .unwrap();
            writer.add_entry(entry(600, None)).unwrap();
            writer.add_entry(entry(100, None)).unwrap();
            writer.add_entry(entry(100, None)).unwrap();

            let data_file_manifest = writer.write_manifest_file().await.unwrap();

            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![data_file_manifest].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }

        /// Writes identical Parquet data files (1.parquet, 2.parquet, 3.parquet)
        /// and returns the file size in bytes.
        fn write_parquet_data_files(&self) -> u64 {
            std::fs::create_dir_all(&self.table_location).unwrap();

            let schema = {
                let fields = vec![
                    arrow_schema::Field::new("x", arrow_schema::DataType::Int64, false)
                        .with_metadata(HashMap::from([(
                            PARQUET_FIELD_ID_META_KEY.to_string(),
                            "1".to_string(),
                        )])),
                    arrow_schema::Field::new("y", arrow_schema::DataType::Int64, false)
                        .with_metadata(HashMap::from([(
                            PARQUET_FIELD_ID_META_KEY.to_string(),
                            "2".to_string(),
                        )])),
                    arrow_schema::Field::new("z", arrow_schema::DataType::Int64, false)
                        .with_metadata(HashMap::from([(
                            PARQUET_FIELD_ID_META_KEY.to_string(),
                            "3".to_string(),
                        )])),
                    arrow_schema::Field::new("a", arrow_schema::DataType::Utf8, false)
                        .with_metadata(HashMap::from([(
                            PARQUET_FIELD_ID_META_KEY.to_string(),
                            "4".to_string(),
                        )])),
                    arrow_schema::Field::new("dbl", arrow_schema::DataType::Float64, false)
                        .with_metadata(HashMap::from([(
                            PARQUET_FIELD_ID_META_KEY.to_string(),
                            "5".to_string(),
                        )])),
                    arrow_schema::Field::new("i32", arrow_schema::DataType::Int32, false)
                        .with_metadata(HashMap::from([(
                            PARQUET_FIELD_ID_META_KEY.to_string(),
                            "6".to_string(),
                        )])),
                    arrow_schema::Field::new("i64", arrow_schema::DataType::Int64, false)
                        .with_metadata(HashMap::from([(
                            PARQUET_FIELD_ID_META_KEY.to_string(),
                            "7".to_string(),
                        )])),
                    arrow_schema::Field::new("bool", arrow_schema::DataType::Boolean, false)
                        .with_metadata(HashMap::from([(
                            PARQUET_FIELD_ID_META_KEY.to_string(),
                            "8".to_string(),
                        )])),
                ];
                Arc::new(arrow_schema::Schema::new(fields))
            };
            // x: [1, 1, 1, 1, ...]
            let col1 = Arc::new(Int64Array::from_iter_values(vec![1; 1024])) as ArrayRef;

            let mut values = vec![2; 512];
            values.append(vec![3; 200].as_mut());
            values.append(vec![4; 300].as_mut());
            values.append(vec![5; 12].as_mut());

            // y: [2, 2, 2, 2, ..., 3, 3, 3, 3, ..., 4, 4, 4, 4, ..., 5, 5, 5, 5]
            let col2 = Arc::new(Int64Array::from_iter_values(values)) as ArrayRef;

            let mut values = vec![3; 512];
            values.append(vec![4; 512].as_mut());

            // z: [3, 3, 3, 3, ..., 4, 4, 4, 4]
            let col3 = Arc::new(Int64Array::from_iter_values(values)) as ArrayRef;

            // a: ["Apache", "Apache", "Apache", ..., "Iceberg", "Iceberg", "Iceberg"]
            let mut values = vec!["Apache"; 512];
            values.append(vec!["Iceberg"; 512].as_mut());
            let col4 = Arc::new(StringArray::from_iter_values(values)) as ArrayRef;

            // dbl:
            let mut values = vec![100.0f64; 512];
            values.append(vec![150.0f64; 12].as_mut());
            values.append(vec![200.0f64; 500].as_mut());
            let col5 = Arc::new(Float64Array::from_iter_values(values)) as ArrayRef;

            // i32:
            let mut values = vec![100i32; 512];
            values.append(vec![150i32; 12].as_mut());
            values.append(vec![200i32; 500].as_mut());
            let col6 = Arc::new(Int32Array::from_iter_values(values)) as ArrayRef;

            // i64:
            let mut values = vec![100i64; 512];
            values.append(vec![150i64; 12].as_mut());
            values.append(vec![200i64; 500].as_mut());
            let col7 = Arc::new(Int64Array::from_iter_values(values)) as ArrayRef;

            // bool:
            let mut values = vec![false; 512];
            values.append(vec![true; 512].as_mut());
            let values: BooleanArray = values.into();
            let col8 = Arc::new(values) as ArrayRef;

            let to_write = RecordBatch::try_new(schema.clone(), vec![
                col1, col2, col3, col4, col5, col6, col7, col8,
            ])
            .unwrap();

            // Write the Parquet files
            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build();

            for n in 1..=3 {
                let file = File::create(format!("{}/{}.parquet", &self.table_location, n)).unwrap();
                let mut writer =
                    ArrowWriter::try_new(file, to_write.schema(), Some(props.clone())).unwrap();

                writer.write(&to_write).expect("Writing batch");

                // writer must be closed to write footer
                writer.close().unwrap();
            }

            std::fs::metadata(format!("{}/1.parquet", &self.table_location))
                .unwrap()
                .len()
        }

        pub async fn setup_unpartitioned_manifest_files(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let parent_snapshot = current_snapshot
                .parent_snapshot(self.table.metadata())
                .unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let current_partition_spec = Arc::new(PartitionSpec::unpartition_spec());

            // Write the data files first, then use the file size in the manifest entries
            let parquet_file_size = self.write_parquet_data_files();

            // Write data files using an empty partition for unpartitioned tables.
            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                current_partition_spec.as_ref().clone(),
            )
            .build_v2_data();

            // Create an empty partition value.
            let empty_partition = Struct::empty();

            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/1.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                .partition(empty_partition.clone())
                                .key_metadata(None)
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();

            writer
                .add_delete_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Deleted)
                        .snapshot_id(parent_snapshot.snapshot_id())
                        .sequence_number(parent_snapshot.sequence_number())
                        .file_sequence_number(parent_snapshot.sequence_number())
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/2.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                .partition(empty_partition.clone())
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();

            writer
                .add_existing_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Existing)
                        .snapshot_id(parent_snapshot.snapshot_id())
                        .sequence_number(parent_snapshot.sequence_number())
                        .file_sequence_number(parent_snapshot.sequence_number())
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/3.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(1)
                                .partition(empty_partition.clone())
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();

            let data_file_manifest = writer.write_manifest_file().await.unwrap();

            // Write to manifest list
            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![data_file_manifest].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }

        /// Writes `1.parquet` as an ID-LESS parquet (no `PARQUET:field_id` field metadata — the
        /// `add_files` migration shape) whose PHYSICAL column order is REVERSED relative to the
        /// table schema: physical column 0 is `y`, column 1 is `x`, both `Int64`. Values are
        /// `x = [1, 1, 1, 1]` and `y = [20, 30, 40, 50]`, chosen so the two columns are trivially
        /// distinguishable. A position-based fallback (which assigns physical col 0 → field id 1 =
        /// `x`) would read `y`'s values into `x` — provably WRONG; only the name mapping (`x`→1,
        /// `y`→2) recovers the correct columns. Returns the file size for the manifest entry.
        fn write_id_less_reversed_parquet(&self) -> u64 {
            std::fs::create_dir_all(&self.table_location).unwrap();

            // NB: deliberately NO PARQUET_FIELD_ID_META_KEY on any field, so the reader takes its
            // "missing field ids" branch (name mapping when present, positional fallback otherwise).
            let schema = Arc::new(arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("y", arrow_schema::DataType::Int64, false),
                arrow_schema::Field::new("x", arrow_schema::DataType::Int64, false),
            ]));

            let y = Arc::new(Int64Array::from_iter_values(vec![20, 30, 40, 50])) as ArrayRef;
            let x = Arc::new(Int64Array::from_iter_values(vec![1, 1, 1, 1])) as ArrayRef;
            let to_write = RecordBatch::try_new(schema.clone(), vec![y, x]).unwrap();

            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build();
            let file = File::create(format!("{}/1.parquet", &self.table_location)).unwrap();
            let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
            writer.write(&to_write).expect("Writing batch");
            writer.close().unwrap();

            std::fs::metadata(format!("{}/1.parquet", &self.table_location))
                .unwrap()
                .len()
        }

        /// Sets up a single-entry unpartitioned manifest over the ID-less reversed
        /// `1.parquet` written by [`Self::write_id_less_reversed_parquet`]. The end-to-end
        /// name-mapping read pin plans this and asserts the columns land correctly.
        pub async fn setup_name_mapping_manifest_files(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let current_partition_spec = Arc::new(PartitionSpec::unpartition_spec());

            let parquet_file_size = self.write_id_less_reversed_parquet();

            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                current_partition_spec.as_ref().clone(),
            )
            .build_v2_data();

            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/1.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(parquet_file_size)
                                .record_count(4)
                                .partition(Struct::empty())
                                .key_metadata(None)
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();

            let data_file_manifest = writer.write_manifest_file().await.unwrap();

            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![data_file_manifest].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }

        pub async fn setup_deadlock_manifests(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let _parent_snapshot = current_snapshot
                .parent_snapshot(self.table.metadata())
                .unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let current_partition_spec = self.table.metadata().default_partition_spec();

            // 1. Write DATA manifest with MULTIPLE entries to fill buffer
            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                current_partition_spec.as_ref().clone(),
            )
            .build_v2_data();

            // Add 10 data entries
            for i in 0..10 {
                writer
                    .add_entry(
                        ManifestEntry::builder()
                            .status(ManifestStatus::Added)
                            .data_file(
                                DataFileBuilder::default()
                                    .partition_spec_id(0)
                                    .content(DataContentType::Data)
                                    .file_path(format!("{}/{}.parquet", &self.table_location, i))
                                    .file_format(DataFileFormat::Parquet)
                                    .file_size_in_bytes(100)
                                    .record_count(1)
                                    .partition(Struct::from_iter([Some(Literal::long(100))]))
                                    .key_metadata(None)
                                    .build()
                                    .unwrap(),
                            )
                            .build(),
                    )
                    .unwrap();
            }
            let data_manifest = writer.write_manifest_file().await.unwrap();

            // 2. Write DELETE manifest
            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                current_partition_spec.as_ref().clone(),
            )
            .build_v2_deletes();

            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::PositionDeletes)
                                .file_path(format!("{}/del.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(100)
                                .record_count(1)
                                .partition(Struct::from_iter([Some(Literal::long(100))]))
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            let delete_manifest = writer.write_manifest_file().await.unwrap();

            // Write to manifest list - DATA FIRST then DELETE
            // This order is crucial for reproduction
            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![data_manifest, delete_manifest].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }

        /// Writes TWO data manifests in DISTINCT identity(x) partitions — one file in
        /// partition `x == 1` (`p1.parquet`) and one in partition `x == 2` (`p2.parquet`),
        /// each in its own manifest with a distinct, asserted file size. A scan filtered to
        /// `x == 1` PRUNES the `x == 2` manifest (its file must not appear in the result),
        /// which is exactly the `skipped_data_manifests` path. The parquet files are never
        /// read by `plan_files`, so only the manifest metadata matters here.
        ///
        /// Returns `(size_partition_one, size_partition_two)` — the per-manifest data file
        /// sizes, so a test can assert `total_file_size_in_bytes` exactly.
        pub async fn setup_two_data_manifests_distinct_partitions(&mut self) -> (u64, u64) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let current_partition_spec = self.table.metadata().default_partition_spec();

            let size_partition_one = 111u64;
            let size_partition_two = 222u64;

            let write_single_file_manifest =
                |partition_value: i64, file_name: &'static str, file_size: u64| {
                    let writer = ManifestWriterBuilder::new(
                        self.next_manifest_file(),
                        Some(current_snapshot.snapshot_id()),
                        None,
                        current_schema.clone(),
                        current_partition_spec.as_ref().clone(),
                    )
                    .build_v2_data();
                    let path = format!("{}/{}", &self.table_location, file_name);
                    (writer, partition_value, path, file_size)
                };

            let mut manifests = vec![];
            for (mut writer, partition_value, path, file_size) in [
                write_single_file_manifest(1, "p1.parquet", size_partition_one),
                write_single_file_manifest(2, "p2.parquet", size_partition_two),
            ] {
                writer
                    .add_entry(
                        ManifestEntry::builder()
                            .status(ManifestStatus::Added)
                            .data_file(
                                DataFileBuilder::default()
                                    .partition_spec_id(0)
                                    .content(DataContentType::Data)
                                    .file_path(path)
                                    .file_format(DataFileFormat::Parquet)
                                    .file_size_in_bytes(file_size)
                                    .record_count(1)
                                    .partition(Struct::from_iter([Some(Literal::long(
                                        partition_value,
                                    ))]))
                                    .key_metadata(None)
                                    .build()
                                    .unwrap(),
                            )
                            .build(),
                    )
                    .unwrap();
                manifests.push(writer.write_manifest_file().await.unwrap());
            }

            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(manifests.into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();

            (size_partition_one, size_partition_two)
        }
    }

    #[test]
    fn test_table_scan_columns() {
        let table = TableTestFixture::new().table;

        let table_scan = table.scan().select(["x", "y"]).build().unwrap();
        assert_eq!(
            Some(vec!["x".to_string(), "y".to_string()]),
            table_scan.column_names
        );

        let table_scan = table
            .scan()
            .select(["x", "y"])
            .select(["z"])
            .build()
            .unwrap();
        assert_eq!(Some(vec!["z".to_string()]), table_scan.column_names);
    }

    #[test]
    fn test_select_all() {
        let table = TableTestFixture::new().table;

        let table_scan = table.scan().select_all().build().unwrap();
        assert!(table_scan.column_names.is_none());
    }

    #[test]
    fn test_select_no_exist_column() {
        let table = TableTestFixture::new().table;

        let table_scan = table.scan().select(["x", "y", "z", "a", "b"]).build();
        assert!(table_scan.is_err());
    }

    #[test]
    fn test_table_scan_default_snapshot_id() {
        let table = TableTestFixture::new().table;

        let table_scan = table.scan().build().unwrap();
        assert_eq!(
            table.metadata().current_snapshot().unwrap().snapshot_id(),
            table_scan.snapshot().unwrap().snapshot_id()
        );
    }

    #[test]
    fn test_table_scan_non_exist_snapshot_id() {
        let table = TableTestFixture::new().table;

        let table_scan = table.scan().snapshot_id(1024).build();
        assert!(table_scan.is_err());
    }

    #[test]
    fn test_table_scan_with_snapshot_id() {
        let table = TableTestFixture::new().table;

        let table_scan = table
            .scan()
            .snapshot_id(3051729675574597004)
            .with_row_selection_enabled(true)
            .build()
            .unwrap();
        assert_eq!(
            table_scan.snapshot().unwrap().snapshot_id(),
            3051729675574597004
        );
    }

    // The shared `example_table_metadata_v2.json` fixture carries two snapshots
    // and a `test` TAG pointing at the OLDER one — exactly the setup needed to
    // prove `use_ref` resolves a reference to a specific (non-current) snapshot.
    /// Current snapshot id (also what the auto-injected `main` ref points at).
    const CURRENT_SNAPSHOT_ID: i64 = 3055729675574597004;
    /// Older snapshot id the `test` tag references.
    const TAG_TEST_SNAPSHOT_ID: i64 = 3051729675574597004;

    #[test]
    fn test_table_scan_use_ref_main_resolves_to_current_snapshot() {
        // RISK: `use_ref("main")` must resolve to the table default (current
        // snapshot), matching Java `useRef(MAIN_BRANCH)` — same as a plain build.
        let table = TableTestFixture::new().table;

        let table_scan = table.scan().use_ref("main").build().unwrap();
        assert_eq!(
            table_scan.snapshot().unwrap().snapshot_id(),
            CURRENT_SNAPSHOT_ID
        );
        assert_eq!(
            table_scan.snapshot().unwrap().snapshot_id(),
            table.metadata().current_snapshot().unwrap().snapshot_id()
        );
    }

    #[test]
    fn test_table_scan_use_ref_tag_resolves_to_referenced_snapshot() {
        // RISK (CORE): a ref pointing at a DIFFERENT snapshot must scan THAT
        // snapshot, not the current one. The `test` tag points at the older
        // snapshot; resolving it must NOT fall back to the current snapshot.
        let table = TableTestFixture::new().table;

        let table_scan = table.scan().use_ref("test").build().unwrap();
        assert_eq!(
            table_scan.snapshot().unwrap().snapshot_id(),
            TAG_TEST_SNAPSHOT_ID
        );
        assert_ne!(
            table_scan.snapshot().unwrap().snapshot_id(),
            table.metadata().current_snapshot().unwrap().snapshot_id()
        );
    }

    #[test]
    fn test_table_scan_use_ref_unknown_ref_errors() {
        // RISK: an unknown ref name must error (Java "Cannot find ref %s"), NOT
        // silently fall back to the current snapshot. The name appears in the
        // message.
        let table = TableTestFixture::new().table;

        let result = table.scan().use_ref("does_not_exist").build();
        let error = result.expect_err("unknown ref should error");
        assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
        assert!(
            error.message().contains("does_not_exist"),
            "error should name the missing ref, got: {error}"
        );
    }

    #[test]
    fn test_table_scan_use_ref_and_snapshot_id_both_set_errors() {
        // RISK: a ref AND an explicit snapshot id are contradictory selectors
        // (Java "Cannot override ref, already set snapshot id") — reject, do not
        // silently let one win.
        let table = TableTestFixture::new().table;

        let result = table
            .scan()
            .use_ref("test")
            .snapshot_id(CURRENT_SNAPSHOT_ID)
            .build();
        let error = result.expect_err("ref + snapshot id should error");
        assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);

        // Order-independent: setting the snapshot id first must reject too.
        let result = table
            .scan()
            .snapshot_id(CURRENT_SNAPSHOT_ID)
            .use_ref("test")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_table_scan_default_without_use_ref_unchanged() {
        // RISK: not calling `use_ref` must leave the default (current snapshot)
        // resolution completely unchanged.
        let table = TableTestFixture::new().table;

        let table_scan = table.scan().build().unwrap();
        assert_eq!(
            table_scan.snapshot().unwrap().snapshot_id(),
            table.metadata().current_snapshot().unwrap().snapshot_id()
        );
    }

    #[tokio::test]
    async fn test_plan_files_use_ref_main_equivalent_to_default_scan() {
        // RISK: `use_ref("main")` must plan the SAME file set as a plain default
        // scan (the current snapshot) — proving the resolved snapshot flows
        // through the full planning pipeline identically.
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        async fn collect_sorted_paths(scan: super::TableScan) -> Vec<String> {
            let tasks: Vec<FileScanTask> = scan
                .plan_files()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let mut paths: Vec<String> =
                tasks.into_iter().map(|task| task.data_file_path).collect();
            paths.sort();
            paths
        }

        let default_paths = collect_sorted_paths(fixture.table.scan().build().unwrap()).await;
        let use_ref_paths =
            collect_sorted_paths(fixture.table.scan().use_ref("main").build().unwrap()).await;

        assert_eq!(default_paths, use_ref_paths);
        assert_eq!(use_ref_paths.len(), 2);
    }

    #[tokio::test]
    async fn test_plan_files_use_non_main_ref_plans_referenced_snapshot() {
        // RISK: a NON-main ref's resolved snapshot must flow through the FULL
        // planning pipeline, not just snapshot-id resolution. Guards against any
        // future special-casing of "main" in build()/planning — a custom `tag`
        // pointing at the current snapshot must plan the same files as the
        // default scan. (The fixture's `test` tag points at the OLDER snapshot,
        // whose manifest list isn't written, so it can't be planned directly;
        // here we add a tag at the CURRENT snapshot, whose manifest list IS
        // written by `setup_manifest_files`.)
        let mut fixture = TableTestFixture::new();

        // Add a non-main tag pointing at the current snapshot, then rebuild the
        // table so the fixture's planning setup runs against the augmented refs.
        let current_snapshot_id = fixture
            .table
            .metadata()
            .current_snapshot()
            .unwrap()
            .snapshot_id();
        let metadata_with_tag = fixture
            .table
            .metadata()
            .clone()
            .into_builder(Some("metadata/v1.json".to_string()))
            .set_ref("at_current", SnapshotReference {
                snapshot_id: current_snapshot_id,
                retention: SnapshotRetention::Tag {
                    max_ref_age_ms: None,
                },
            })
            .unwrap()
            .build()
            .unwrap()
            .metadata;
        fixture.table = Table::builder()
            .metadata(metadata_with_tag)
            .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
            .file_io(fixture.table.file_io().clone())
            .metadata_location(format!("{}/metadata/v1.json", &fixture.table_location))
            .build()
            .unwrap();

        fixture.setup_manifest_files().await;

        async fn collect_sorted_paths(scan: super::TableScan) -> Vec<String> {
            let tasks: Vec<FileScanTask> = scan
                .plan_files()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let mut paths: Vec<String> =
                tasks.into_iter().map(|task| task.data_file_path).collect();
            paths.sort();
            paths
        }

        let default_paths = collect_sorted_paths(fixture.table.scan().build().unwrap()).await;
        let use_ref_paths =
            collect_sorted_paths(fixture.table.scan().use_ref("at_current").build().unwrap()).await;

        assert_eq!(default_paths, use_ref_paths);
        assert_eq!(use_ref_paths.len(), 2);
    }

    #[tokio::test]
    async fn test_plan_files_on_table_without_any_snapshots() {
        let table = TableTestFixture::new_empty().table;
        let batch_stream = table.scan().build().unwrap().to_arrow().await.unwrap();
        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();
        assert!(batches.is_empty());
    }

    #[tokio::test]
    async fn test_plan_files_no_deletions() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Create table scan for current snapshot and plan files
        let table_scan = fixture
            .table
            .scan()
            .with_row_selection_enabled(true)
            .build()
            .unwrap();

        let mut tasks = table_scan
            .plan_files()
            .await
            .unwrap()
            .try_fold(vec![], |mut acc, task| async move {
                acc.push(task);
                Ok(acc)
            })
            .await
            .unwrap();

        assert_eq!(tasks.len(), 2);

        tasks.sort_by_key(|t| t.data_file_path.to_string());

        // Check first task is added data file
        assert_eq!(
            tasks[0].data_file_path,
            format!("{}/1.parquet", &fixture.table_location)
        );

        // Check second task is existing data file
        assert_eq!(
            tasks[1].data_file_path,
            format!("{}/3.parquet", &fixture.table_location)
        );
    }

    // =======================================================================================
    // plan_tasks() — split + bin-pack grouping (Java Scan.planTasks()).
    // =======================================================================================

    /// The total byte length across all sub-tasks of all groups — the conservation quantity that
    /// must equal the sum of the source files' declared sizes (the split tiles every file with no
    /// gap/overlap).
    fn total_group_length(groups: &[CombinedScanTask]) -> u64 {
        groups
            .iter()
            .flat_map(|group| group.tasks().iter())
            .map(|task| task.length)
            .sum()
    }

    /// End-to-end: the planning fixture (files 1000+600+100+100 = 1800 declared bytes; the 1000-byte
    /// file carries split offsets [0,400,800]) plans into groups whose per-group weight is within
    /// the target and whose total length is conserved.
    #[tokio::test]
    async fn test_plan_tasks_groups_conserve_length_and_respect_target() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_for_planning().await;

        // target=500, lookback large, open-file-cost=0 so the weight is purely the byte length
        // (isolates the split + pack from the open-cost floor).
        let table_scan = fixture
            .table
            .scan()
            .with_split_size(500)
            .with_split_lookback(10)
            .with_split_open_file_cost(0)
            .build()
            .unwrap();

        let groups: Vec<CombinedScanTask> = table_scan
            .plan_tasks()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        // Length conservation: every byte of every file is covered exactly once.
        assert_eq!(
            total_group_length(&groups),
            1800,
            "the split must tile all four files (1000+600+100+100) exactly"
        );

        // With open-file-cost 0 the weight == group length, so every group must be <= target
        // UNLESS it is a single oversized sub-task (none here: max sub-task is the 600-byte file's
        // 500-byte first window, still <= 500... actually fixed-size keeps each <= target).
        for group in &groups {
            let weight: u64 = group.tasks().iter().map(|t| t.length).sum();
            assert!(
                weight <= 500 || group.files_count() == 1,
                "group weight {weight} exceeds target 500 and is not a lone oversized task"
            );
            // size_bytes is the sum of member lengths (not the weight).
            assert_eq!(group.size_bytes(), weight);
        }

        // The 1000-byte file is offsets-aware split into windows (0,400)(400,400)(800,200); the
        // 600-byte file is fixed-size split into (0,500)(500,100); the two 100-byte files are whole.
        // ⇒ 3 + 2 + 1 + 1 = 7 sub-tasks total across all groups.
        let total_sub_tasks: usize = groups.iter().map(|g| g.files_count()).sum();
        assert_eq!(total_sub_tasks, 7, "expected 7 split sub-tasks");
    }

    /// A LARGE target collapses everything into a single group (one bin holds all 1800 bytes).
    #[tokio::test]
    async fn test_plan_tasks_large_target_single_group() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_for_planning().await;

        let table_scan = fixture
            .table
            .scan()
            .with_split_size(1_000_000)
            .with_split_open_file_cost(0)
            .build()
            .unwrap();

        let groups: Vec<CombinedScanTask> = table_scan
            .plan_tasks()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        // The 1000-byte file's offsets still split it into 3 windows (offsets-aware IGNORES target),
        // but they all pack into ONE bin with everything else.
        assert_eq!(
            groups.len(),
            1,
            "a huge target packs all sub-tasks into one group"
        );
        assert_eq!(total_group_length(&groups), 1800);
    }

    /// The open-file-cost floor inflates each task's weight, forcing MORE groups than the raw byte
    /// length would. Mutation bait: if the weight ignored the `(1 + #deletes) * openFileCost` floor,
    /// the group count would collapse to the length-only case.
    #[tokio::test]
    async fn test_plan_tasks_open_file_cost_floor_changes_grouping() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_for_planning().await;

        // target = 500, open-file-cost = 500. Each whole/sub-task now weighs at least 500 (the
        // floor for a delete-free task is 1 * 500), so NO two sub-tasks can share a bin — every
        // sub-task lands in its own group.
        let table_scan = fixture
            .table
            .scan()
            .with_split_size(500)
            .with_split_lookback(10)
            .with_split_open_file_cost(500)
            .build()
            .unwrap();

        let groups: Vec<CombinedScanTask> = table_scan
            .plan_tasks()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        // 7 sub-tasks, each its own group because the floor (>=500) fills the 500 target alone.
        assert_eq!(
            groups.len(),
            7,
            "the open-file-cost floor must force one group per sub-task (mutation bait)"
        );
        assert_eq!(total_group_length(&groups), 1800);
    }

    /// `plan_tasks` rejects an invalid split target (Java `validatePlanningArguments`).
    #[tokio::test]
    async fn test_plan_tasks_rejects_zero_lookback() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_for_planning().await;

        let table_scan = fixture
            .table
            .scan()
            .with_split_size(500)
            .with_split_lookback(0)
            .build()
            .unwrap();

        assert!(
            table_scan.plan_tasks().await.is_err(),
            "lookback 0 must be rejected"
        );
    }

    /// `plan_tasks` over a snapshotless table is an empty stream (no panic).
    #[tokio::test]
    async fn test_plan_tasks_empty_when_no_snapshot() {
        let fixture = TableTestFixture::new_empty();
        let table_scan = fixture.table.scan().build().unwrap();
        let groups: Vec<CombinedScanTask> = table_scan
            .plan_tasks()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert!(groups.is_empty());
    }

    /// The builder knobs resolve from table properties when no scan-time override is given, and the
    /// override wins over the property (Java `BaseScan.targetSplitSize` precedence). Pinned by the
    /// resulting group count differing between the property default and an override.
    #[tokio::test]
    async fn test_plan_tasks_split_size_override_wins_over_property() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_for_planning().await;

        // Override target to a tiny 200 bytes with no open-file-cost ⇒ many small groups.
        let scan_small = fixture
            .table
            .scan()
            .with_split_size(200)
            .with_split_open_file_cost(0)
            .build()
            .unwrap();
        let small: Vec<CombinedScanTask> = scan_small
            .plan_tasks()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        // A big override target ⇒ fewer (one) groups. The override must change the outcome.
        let scan_big = fixture
            .table
            .scan()
            .with_split_size(1_000_000)
            .with_split_open_file_cost(0)
            .build()
            .unwrap();
        let big: Vec<CombinedScanTask> = scan_big
            .plan_tasks()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        assert!(
            small.len() > big.len(),
            "a smaller split target must yield more groups (override is load-bearing): {} vs {}",
            small.len(),
            big.len()
        );
        // Length is conserved regardless of target.
        assert_eq!(total_group_length(&small), 1800);
        assert_eq!(total_group_length(&big), 1800);
    }

    /// Drains a `FileScanTask` stream into a path-sorted `Vec`, asserting every item is
    /// `Ok`. Used by the metrics-emission tests so they consume the WHOLE stream (the
    /// trigger for a `ScanReport`).
    async fn drain_sorted(stream: FileScanTaskStream) -> Vec<FileScanTask> {
        let mut tasks: Vec<FileScanTask> = stream.try_collect().await.unwrap();
        tasks.sort_by_key(|task| task.data_file_path.to_string());
        tasks
    }

    /// Extracts the single `i64` counter value for a metric, panicking if the metric is
    /// absent — a sharp failure when an expected counter was left `None`.
    fn counter(value: Option<crate::metrics::CounterResult>, metric: &str) -> i64 {
        value
            .unwrap_or_else(|| panic!("metric {metric} should be present"))
            .value
    }

    /// Risk (TEST 1 / mutation d): installing the metrics machinery silently changes the
    /// planned task set even when NO reporter is configured. Pins the default (opt-in-OFF)
    /// scan plans EXACTLY the same tasks as before — the byte-unchanged guarantee.
    #[tokio::test]
    async fn test_plan_files_without_reporter_is_unchanged() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let table_scan = fixture.table.scan().build().unwrap();
        let tasks = drain_sorted(table_scan.plan_files().await.unwrap()).await;

        // Identical to `test_plan_files_no_deletions`: 1.parquet (Added) + 3.parquet
        // (Existing); the 2.parquet Deleted tombstone is filtered out.
        assert_eq!(tasks.len(), 2);
        assert_eq!(
            tasks[0].data_file_path,
            format!("{}/1.parquet", &fixture.table_location)
        );
        assert_eq!(
            tasks[1].data_file_path,
            format!("{}/3.parquet", &fixture.table_location)
        );
    }

    /// Risk (TEST 1 / mutation d, STRUCTURAL): the opt-in is broken — a collector (and its
    /// per-manifest/per-task counting overhead) is installed even when NO reporter is
    /// configured. Pins that a default scan threads `None` into the plan context (no
    /// collector) and `Some` only when a reporter is set. This is the perf/scope guard the
    /// task-set test alone cannot catch (counting does not change which tasks are planned).
    #[tokio::test]
    async fn test_no_reporter_means_no_collector_installed() {
        let fixture = TableTestFixture::new();

        let scan_without_reporter = fixture.table.scan().build().unwrap();
        assert!(
            scan_without_reporter.metrics.is_none(),
            "no reporter ⇒ no metrics context on the scan"
        );
        assert!(
            scan_without_reporter
                .plan_context
                .as_ref()
                .unwrap()
                .metrics_collector
                .is_none(),
            "no reporter ⇒ no collector threaded into planning (opt-in)"
        );

        let reporter = Arc::new(InMemoryMetricsReporter::new());
        let scan_with_reporter = fixture
            .table
            .scan()
            .with_metrics_reporter(reporter)
            .build()
            .unwrap();
        assert!(
            scan_with_reporter
                .plan_context
                .as_ref()
                .unwrap()
                .metrics_collector
                .is_some(),
            "a reporter ⇒ a collector is installed"
        );
    }

    /// Risk (TEST 2): the emitted report carries the wrong table / snapshot / filter /
    /// projection, or inaccurate counters (data-file count, manifest totals, summed bytes).
    /// Pins ONE report with the right identity AND every cleanly-collected counter exact.
    #[tokio::test]
    async fn test_scan_report_emitted_with_accurate_counters_on_full_consumption() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let parquet_file_size = std::fs::metadata(format!("{}/1.parquet", &fixture.table_location))
            .unwrap()
            .len() as i64;
        let snapshot_id = fixture
            .table
            .metadata()
            .current_snapshot()
            .unwrap()
            .snapshot_id();

        let reporter = Arc::new(InMemoryMetricsReporter::new());
        let table_scan = fixture
            .table
            .scan()
            .with_metrics_reporter(reporter.clone())
            .build()
            .unwrap();

        // No report until the stream is consumed.
        assert!(reporter.last_report().is_none(), "no report before consume");

        let tasks = drain_sorted(table_scan.plan_files().await.unwrap()).await;
        assert_eq!(tasks.len(), 2, "1.parquet + 3.parquet, tombstone dropped");

        let report = reporter
            .last_scan_report()
            .expect("a scan report after full consumption");

        // Report identity.
        assert_eq!(report.table_name, "db.table1");
        assert_eq!(report.snapshot_id, snapshot_id);
        // No scan filter ⇒ the report's filter is `alwaysTrue` (Java `BaseScan.filter()`).
        assert_eq!(report.filter, Predicate::AlwaysTrue);
        // The default projection is all 8 columns of the fixture schema.
        assert_eq!(report.projected_field_ids.len(), 8);
        assert!(report.projected_field_names.contains(&"x".to_string()));

        let metrics = &report.scan_metrics;
        // 1 data manifest, 0 delete manifests in the fixture's list.
        assert_eq!(
            counter(metrics.total_data_manifests.clone(), "total-data"),
            1
        );
        assert_eq!(
            counter(metrics.total_delete_manifests.clone(), "total-delete"),
            0
        );
        // No filter ⇒ the single data manifest is scanned, none skipped.
        assert_eq!(
            counter(metrics.scanned_data_manifests.clone(), "scanned"),
            1
        );
        assert_eq!(
            counter(metrics.skipped_data_manifests.clone(), "skipped"),
            0
        );
        // result-data-files == # tasks; no deletes attached.
        assert_eq!(counter(metrics.result_data_files.clone(), "result-data"), 2);
        assert_eq!(
            counter(metrics.result_delete_files.clone(), "result-delete"),
            0
        );
        // total bytes == sum of the two produced data files' sizes.
        assert_eq!(
            counter(metrics.total_file_size_in_bytes.clone(), "total-bytes"),
            2 * parquet_file_size
        );
        // The planning timer is populated (1 timed event).
        let timer = metrics
            .total_planning_duration
            .clone()
            .expect("planning duration present");
        assert_eq!(timer.count, 1);
    }

    /// Risk (TEST 3 / mutation a): a pruned manifest is miscounted (as scanned instead of
    /// skipped), or its file leaks into the result. Pins a filtered scan that PRUNES the
    /// `x == 2` manifest: `skipped_data_manifests >= 1`, the pruned file is absent, and the
    /// surviving file's bytes are the ONLY bytes summed.
    #[tokio::test]
    async fn test_filtered_scan_reports_pruned_manifest_as_skipped() {
        let mut fixture = TableTestFixture::new();
        let (size_partition_one, _size_partition_two) =
            fixture.setup_two_data_manifests_distinct_partitions().await;

        let reporter = Arc::new(InMemoryMetricsReporter::new());
        let table_scan = fixture
            .table
            .scan()
            .with_filter(Reference::new("x").equal_to(Datum::long(1)))
            .with_metrics_reporter(reporter.clone())
            .build()
            .unwrap();

        let tasks = drain_sorted(table_scan.plan_files().await.unwrap()).await;

        // Only the `x == 1` file survives; the `x == 2` file's manifest was pruned.
        assert_eq!(tasks.len(), 1);
        assert_eq!(
            tasks[0].data_file_path,
            format!("{}/p1.parquet", &fixture.table_location)
        );
        assert!(
            !tasks
                .iter()
                .any(|task| task.data_file_path.ends_with("p2.parquet")),
            "the pruned x==2 file must not appear in the result"
        );

        let report = reporter.last_scan_report().expect("a scan report");
        let metrics = &report.scan_metrics;

        assert_eq!(counter(metrics.total_data_manifests.clone(), "total"), 2);
        assert_eq!(
            counter(metrics.scanned_data_manifests.clone(), "scanned"),
            1
        );
        assert_eq!(
            counter(metrics.skipped_data_manifests.clone(), "skipped"),
            1,
            "the x==2 manifest is pruned"
        );
        // scanned + skipped == total.
        assert_eq!(
            counter(metrics.scanned_data_manifests.clone(), "s")
                + counter(metrics.skipped_data_manifests.clone(), "k"),
            counter(metrics.total_data_manifests.clone(), "t"),
        );
        assert_eq!(counter(metrics.result_data_files.clone(), "result"), 1);
        // Only the surviving file's bytes are summed (the pruned file's are not).
        assert_eq!(
            counter(metrics.total_file_size_in_bytes.clone(), "bytes"),
            size_partition_one as i64,
        );
    }

    /// Risk (TEST 4 / mutation b): delete manifests and delete-file references are
    /// miscounted (e.g. delete files folded into `result_data_files`, or the delete
    /// manifest not counted). Pins a fixture with a DELETE manifest: `total_delete_manifests`
    /// and `result_delete_files` are counted, and `result_data_files` excludes them.
    #[tokio::test]
    async fn test_delete_manifest_fixture_counts_delete_files_separately() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_deadlock_manifests().await;

        let reporter = Arc::new(InMemoryMetricsReporter::new());
        let table_scan = fixture
            .table
            .scan()
            .with_metrics_reporter(reporter.clone())
            .build()
            .unwrap();

        let tasks = drain_sorted(table_scan.plan_files().await.unwrap()).await;
        // 10 data files, each picking up the single position-delete in their shared
        // partition (x == 100).
        assert_eq!(tasks.len(), 10);
        let total_delete_refs: usize = tasks.iter().map(|task| task.deletes.len()).sum();

        let report = reporter.last_scan_report().expect("a scan report");
        let metrics = &report.scan_metrics;

        assert_eq!(counter(metrics.total_data_manifests.clone(), "data"), 1);
        assert_eq!(
            counter(metrics.total_delete_manifests.clone(), "delete-manifests"),
            1,
            "the one delete manifest is counted"
        );
        // result_data_files counts ONLY the 10 data tasks, not the delete file.
        assert_eq!(
            counter(metrics.result_data_files.clone(), "result-data"),
            10
        );
        // result_delete_files == total delete-file references across the tasks.
        assert_eq!(
            counter(metrics.result_delete_files.clone(), "result-delete"),
            total_delete_refs as i64,
        );
        assert!(
            total_delete_refs >= 1,
            "the delete file must attach to at least one data task"
        );
    }

    /// Risk (TEST 5 / mutation c): the report is emitted PER-TASK (or never). Pins that
    /// fully consuming the stream produces EXACTLY ONE report — by snapshotting the last
    /// report, then confirming a fresh reporter observes the same single emission and that
    /// re-polling the exhausted stream does not re-report.
    #[tokio::test]
    async fn test_scan_report_emitted_exactly_once() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // A counting reporter that records HOW MANY times `report` was called.
        #[derive(Debug, Default)]
        struct CountingReporter {
            count: std::sync::atomic::AtomicUsize,
        }
        impl crate::metrics::MetricsReporter for CountingReporter {
            fn report(&self, _report: MetricsReport) {
                self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let reporter = Arc::new(CountingReporter::default());
        let table_scan = fixture
            .table
            .scan()
            .with_metrics_reporter(reporter.clone())
            .build()
            .unwrap();

        let mut stream = table_scan.plan_files().await.unwrap();
        // Drain task-by-task; the report must NOT fire per task.
        let mut seen = 0;
        while let Some(task) = stream.try_next().await.unwrap() {
            let _ = task;
            seen += 1;
            assert_eq!(
                reporter.count.load(std::sync::atomic::Ordering::SeqCst),
                0,
                "no report should be emitted mid-stream (after {seen} tasks)"
            );
        }

        assert_eq!(seen, 2);
        assert_eq!(
            reporter.count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "exactly one report on full consumption"
        );

        // Re-polling the exhausted stream does not re-report.
        assert!(stream.try_next().await.unwrap().is_none());
        assert_eq!(
            reporter.count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "re-polling an exhausted stream does not re-report"
        );
    }

    /// Risk (EMISSION / early-drop): the report fires with PARTIAL counts when the consumer
    /// abandons the stream before exhausting it — Java only reports on the iterable's CLOSE
    /// after full consumption, and this wrapper emits ONLY on the `Ready(None)` transition.
    /// Pins that pulling a few tasks and then DROPPING the stream emits NO report (the
    /// `Ready(None)` arm is never reached) and does not hang — the spawned planning tasks
    /// must not deadlock when the receiver is dropped early.
    #[tokio::test]
    async fn test_partial_consume_then_drop_emits_no_report() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_deadlock_manifests().await;

        let reporter = Arc::new(InMemoryMetricsReporter::new());
        let table_scan = fixture
            .table
            .scan()
            .with_metrics_reporter(reporter.clone())
            .build()
            .unwrap();

        let mut stream = table_scan.plan_files().await.unwrap();
        // Pull only 3 of the fixture's 10 tasks, then abandon the stream.
        for _ in 0..3 {
            stream
                .try_next()
                .await
                .unwrap()
                .expect("a task should be produced");
        }
        drop(stream);

        // No `Ready(None)` was observed, so no report should have been emitted — a
        // partial-count report would be wrong. (Drop must not hang on the producers.)
        assert!(
            reporter.last_report().is_none(),
            "dropping the stream before exhaustion must NOT emit a (partial) report"
        );
    }

    /// Risk: the byte counters are reported with the wrong unit. Pins the bytes counter on
    /// an emitted report carries `MetricUnit::Bytes` (end-to-end, not just at the collector).
    #[tokio::test]
    async fn test_emitted_byte_counter_carries_bytes_unit() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let reporter = Arc::new(InMemoryMetricsReporter::new());
        let table_scan = fixture
            .table
            .scan()
            .with_metrics_reporter(reporter.clone())
            .build()
            .unwrap();

        let _ = drain_sorted(table_scan.plan_files().await.unwrap()).await;
        let report = reporter.last_scan_report().expect("a scan report");
        assert_eq!(
            report.scan_metrics.total_file_size_in_bytes.unwrap().unit,
            MetricUnit::Bytes
        );
    }

    #[tokio::test]
    async fn test_open_parquet_no_deletions() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Create table scan for current snapshot and plan files
        let table_scan = fixture
            .table
            .scan()
            .with_row_selection_enabled(true)
            .build()
            .unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        let col = batches[0].column_by_name("x").unwrap();

        // `x` is an identity-partition column, materialized by the reader as a plain
        // Int64 constant from the partition metadata (value 1), matching the scan schema.
        let int64_arr = decode_int64_column(col);
        assert_eq!(int64_arr.value(0), 1);
    }

    #[tokio::test]
    async fn test_open_parquet_no_deletions_by_separate_reader() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Create table scan for current snapshot and plan files
        let table_scan = fixture
            .table
            .scan()
            .with_row_selection_enabled(true)
            .build()
            .unwrap();

        let mut plan_task: Vec<_> = table_scan
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(plan_task.len(), 2);

        let reader = ArrowReaderBuilder::new(fixture.table.file_io().clone()).build();
        let batch_stream = reader
            .clone()
            .read(Box::pin(stream::iter(vec![Ok(plan_task.remove(0))])))
            .unwrap();
        let batch_1: Vec<_> = batch_stream.try_collect().await.unwrap();

        let reader = ArrowReaderBuilder::new(fixture.table.file_io().clone()).build();
        let batch_stream = reader
            .read(Box::pin(stream::iter(vec![Ok(plan_task.remove(0))])))
            .unwrap();
        let batch_2: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batch_1, batch_2);
    }

    // ---- name-mapping scan wiring (C-5) ----

    /// The mapping used by the name-mapping pins: `x`→field id 1, `y`→field id 2, matching the
    /// unpartitioned fixture schema. Kebab-case JSON, exactly as Java `NameMappingParser` emits.
    /// `pub` because the incremental-scan pins (`incremental.rs`) reuse it against the same
    /// fixture rather than duplicating the literal.
    pub const NAME_MAPPING_X1_Y2: &str =
        r#"[{"field-id":1,"names":["x"]},{"field-id":2,"names":["y"]}]"#;

    /// C-5(a): a valid `schema.name-mapping.default` property is parsed once per plan and threaded
    /// onto EVERY produced task, with the parsed CONTENT (not merely `is_some`).
    #[tokio::test]
    async fn test_plan_threads_name_mapping_onto_every_task() {
        let mut fixture = TableTestFixture::new_with_name_mapping_property(NAME_MAPPING_X1_Y2);
        fixture.setup_unpartitioned_manifest_files().await;

        let tasks: Vec<FileScanTask> = fixture
            .table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        assert!(!tasks.is_empty(), "fixture must produce at least one task");
        for task in &tasks {
            let mapping = task
                .name_mapping
                .as_ref()
                .expect("every planned task carries the parsed name mapping");
            let fields = mapping.fields();
            assert_eq!(fields.len(), 2, "mapping must have both mapped fields");
            assert_eq!(fields[0].field_id(), Some(1));
            assert_eq!(fields[0].names().to_vec(), vec!["x".to_string()]);
            assert_eq!(fields[1].field_id(), Some(2));
            assert_eq!(fields[1].names().to_vec(), vec!["y".to_string()]);
        }
    }

    /// C-5(b): with NO name-mapping property the tasks carry `None` (the position-fallback path is
    /// left untouched).
    #[tokio::test]
    async fn test_plan_absent_name_mapping_property_yields_none() {
        let mut fixture = TableTestFixture::new_unpartitioned();
        fixture.setup_unpartitioned_manifest_files().await;

        let tasks: Vec<FileScanTask> = fixture
            .table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        assert!(!tasks.is_empty(), "fixture must produce at least one task");
        for task in &tasks {
            assert!(
                task.name_mapping.is_none(),
                "an absent property must leave name_mapping None"
            );
        }
    }

    /// C-5(c) / C-3: an unparsable name-mapping property is a LOUD typed error that NAMES the
    /// property — never a silent `None`. Each partition element Java rejects is pinned: malformed
    /// JSON, an EMPTY value, and a WHITESPACE-only value. Java `NameMappingParser.fromJson(String)`
    /// delegates to `ObjectMapper.readValue`, which throws (wrapped as `UncheckedIOException`) on
    /// all three — Jackson reports "no content" for empty/whitespace — so Rust must fail loud too.
    #[tokio::test]
    async fn test_invalid_name_mapping_property_is_loud_typed_error() {
        // `build()` parses the property (once per plan), so an unparsable value errors there.
        for bad in ["{ this is not a valid name mapping", "", "   "] {
            let fixture = TableTestFixture::new_with_name_mapping_property(bad);

            let err = fixture.table.scan().build().err().unwrap_or_else(|| {
                panic!(
                    "an unparsable name-mapping property ({bad:?}) must fail the scan build loudly"
                )
            });

            assert_eq!(err.kind(), crate::ErrorKind::DataInvalid, "value = {bad:?}");
            assert!(
                err.to_string()
                    .contains(TableProperties::PROPERTY_DEFAULT_NAME_MAPPING),
                "the error must name the offending property (value = {bad:?}), got: {err}"
            );
        }
    }

    /// C-5(d): END-TO-END. An ID-less parquet whose physical column order is REVERSED relative to
    /// the schema scans to the CORRECT columns via the name mapping. This exercises the full
    /// `TableScan` → plan → arrow-read path (not just the reader-level unit), and the mapping is
    /// NON-trivial: a positional fallback would read `y`'s values into `x`, so the exact-value
    /// asserts below go RED if the wiring re-hardcodes `name_mapping: None`.
    #[tokio::test]
    async fn test_scan_applies_name_mapping_to_id_less_parquet() {
        let mut fixture = TableTestFixture::new_with_name_mapping_property(NAME_MAPPING_X1_Y2);
        fixture.setup_name_mapping_manifest_files().await;

        let table_scan = fixture.table.scan().select(["x", "y"]).build().unwrap();

        // Wiring: the single planned task carries the parsed mapping.
        let tasks: Vec<FileScanTask> = table_scan
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(tasks.len(), 1, "single-entry manifest yields one task");
        assert!(
            tasks[0].name_mapping.is_some(),
            "the planned task must carry the name mapping"
        );

        // End-to-end read: the reversed ID-less parquet must map to the CORRECT columns.
        let batches: Vec<RecordBatch> = table_scan
            .to_arrow()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert!(!batches.is_empty(), "scan must return at least one batch");

        let x = decode_int64_column(batches[0].column_by_name("x").unwrap());
        let y = decode_int64_column(batches[0].column_by_name("y").unwrap());

        // Under the name mapping, `x` resolves to the physical `x` column (all 1s). Under a
        // positional fallback `x` would read physical column 0 (`y` = 20,30,40,50) — that contrast
        // is the pin: these exact-value asserts are what a re-hardcoded `None` breaks.
        assert_eq!(
            x.values(),
            &[1, 1, 1, 1],
            "x must be the mapped physical x column (all 1s), not the positional y column"
        );
        assert_eq!(
            y.values(),
            &[20, 30, 40, 50],
            "y must be the mapped physical y column"
        );
    }

    /// C-5(d), subset projection. Projecting ONLY `x` out of the reversed ID-less parquet must
    /// still read the physical `x` column (all 1s), not physical column 0 (`y`). This pins the
    /// FIELD-ID-based projection mask: a positional projection would select physical column 0 by
    /// `field_id - 1`, reading `y` (or, after the transformer reorders by id, producing a NULL
    /// `x`) — so this asserts the reader's projection resolves the subset by field id.
    #[tokio::test]
    async fn test_scan_name_mapping_subset_projection_reads_correct_physical_column() {
        let mut fixture = TableTestFixture::new_with_name_mapping_property(NAME_MAPPING_X1_Y2);
        fixture.setup_name_mapping_manifest_files().await;

        let table_scan = fixture.table.scan().select(["x"]).build().unwrap();
        let batches: Vec<RecordBatch> = table_scan
            .to_arrow()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert!(!batches.is_empty(), "scan must return at least one batch");

        let x = decode_int64_column(batches[0].column_by_name("x").unwrap());
        assert_eq!(
            x.values(),
            &[1, 1, 1, 1],
            "subset-projected x must resolve to the physical x column by field id"
        );
    }

    #[tokio::test]
    async fn test_open_parquet_with_projection() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Create table scan for current snapshot and plan files
        let table_scan = fixture
            .table
            .scan()
            .select(["x", "z"])
            .with_row_selection_enabled(true)
            .build()
            .unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches[0].num_columns(), 2);

        let col1 = batches[0].column_by_name("x").unwrap();
        // `x` is an identity-partition constant materialized from partition metadata
        // (value 1) as a plain Int64 column matching the declared scan schema.
        let int64_arr = decode_int64_column(col1);
        assert_eq!(int64_arr.value(0), 1);

        let col2 = batches[0].column_by_name("z").unwrap();
        let int64_arr = col2.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int64_arr.value(0), 3);

        // test empty scan
        let table_scan = fixture.table.scan().select_empty().build().unwrap();
        let batch_stream = table_scan.to_arrow().await.unwrap();
        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches[0].num_columns(), 0);
        assert_eq!(batches[0].num_rows(), 1024);
    }

    /// The load-bearing semantic of the identity-partition constants map: the value of
    /// an identity-partition column comes from PARTITION METADATA, not the data file.
    ///
    /// The fixture writes a data file whose parquet `x` column is `[1; 1024]` but whose
    /// manifest partition value is `x == 999`. A scan must return 999 (the partition
    /// metadata constant, Java `PartitionUtil.constantsMap`), NOT 1 (the file bytes).
    /// This is the test the whole feature exists for, and the activation MUTATION
    /// (`task.partition_spec = None`) makes it fail by reading 1 from the file.
    ///
    /// The output `x` is also asserted to be a PLAIN `Int64Array` (declared `Int64`,
    /// not RunEndEncoded) — pinning the bug-(a) schema-equality fix end-to-end through
    /// the reader, not only at the transformer unit level.
    #[tokio::test]
    async fn test_identity_partition_column_value_comes_from_metadata_not_file() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files_partition_value_differs().await;

        let table_scan = fixture.table.scan().select(["x", "y"]).build().unwrap();
        let batches: Vec<_> = table_scan
            .to_arrow()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(total_rows, 1024);

        for batch in &batches {
            let x_col = batch.column_by_name("x").unwrap();
            // EXACT physical type: a plain Int64 column (the declared scan-schema type),
            // never RunEndEncoded.
            assert_eq!(
                x_col.data_type(),
                &arrow_schema::DataType::Int64,
                "identity-partition `x` must be a plain Int64 column, not REE"
            );
            let x = x_col.as_any().downcast_ref::<Int64Array>().unwrap();
            for index in 0..x.len() {
                assert_eq!(
                    x.value(index),
                    999,
                    "x must be the PARTITION metadata value (999), not the file value (1)"
                );
            }
            // `y` is NOT a partition column, so it is read from the file unchanged.
            let y = batch.column_by_name("y").unwrap();
            let y = y.as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(y.value(0), 2);
        }
    }

    /// Multi-spec interaction (this increment sits on top of the multi-spec WRITE
    /// increment): each data file's identity-partition constants are materialized under
    /// THAT file's own spec, resolved from its manifest's `partition_spec_id`.
    ///
    /// The fixture writes 1.parquet under `identity(x)` (spec 0, partition `x == 777`)
    /// and 2.parquet under `identity(y)` (spec 1, partition `y == 888`), each in its own
    /// manifest. Both files share the same parquet bytes (`x == 1`, `y` starting at `2`).
    /// A correct scan returns, per file:
    ///  * spec-0 file: x = 777 (constant from spec 0), y read from file (= 2)
    ///  * spec-1 file: y = 888 (constant from spec 1), x read from file (= 1)
    ///
    /// A single-spec scan (using one shared spec for both files) cannot produce this.
    #[tokio::test]
    async fn test_multispec_scan_materializes_each_file_constant_under_its_own_spec() {
        let mut fixture = TableTestFixture::new_with_two_identity_specs();
        fixture.setup_two_identity_spec_manifests().await;

        let table_scan = fixture.table.scan().select(["x", "y"]).build().unwrap();
        let batches: Vec<_> = table_scan
            .to_arrow()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        // Each batch corresponds to one file; identify which spec produced it by its
        // constant column, then assert the OTHER column was read from the file.
        let mut saw_spec_zero_file = false;
        let mut saw_spec_one_file = false;
        for batch in &batches {
            let x = decode_int64_column(batch.column_by_name("x").unwrap());
            let y = decode_int64_column(batch.column_by_name("y").unwrap());

            if x.value(0) == 777 {
                // spec-0 file: x is the constant 777, y read from the file (= 2).
                saw_spec_zero_file = true;
                for index in 0..x.len() {
                    assert_eq!(x.value(index), 777, "spec-0 file: x constant under spec 0");
                }
                assert_eq!(y.value(0), 2, "spec-0 file: y read from file");
            } else {
                // spec-1 file: y is the constant 888, x read from the file (= 1).
                saw_spec_one_file = true;
                for index in 0..y.len() {
                    assert_eq!(y.value(index), 888, "spec-1 file: y constant under spec 1");
                }
                assert_eq!(x.value(0), 1, "spec-1 file: x read from file");
            }
        }
        assert!(
            saw_spec_zero_file && saw_spec_one_file,
            "scan must surface BOTH the spec-0 and spec-1 files (each under its own spec)"
        );
    }

    /// Null identity-partition value, end-to-end through the scan/reader path.
    ///
    /// Per Iceberg column-projection rule #4 (`constants_map` skips a null partition value,
    /// Java `PartitionUtil.constantsMap` only puts non-null converted values), a null
    /// partition value does NOT enter the constants map, so it does not override anything.
    /// In THIS fixture the data file physically contains `x` (`[1; 1024]`), so the column
    /// resolves to the FILE value (1), not an error — the absence of an override is the
    /// behavior under test. (When the file LACKS the column, the same null path yields a
    /// null column — pinned by the transformer unit test `null_identity_partition_value`.)
    /// The point here is that a null partition value is handled WITHOUT erroring and the
    /// column keeps its plain declared scan-schema type.
    #[tokio::test]
    async fn test_identity_partition_null_value_does_not_override_and_does_not_error() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files_null_partition().await;

        let table_scan = fixture.table.scan().select(["x", "y"]).build().unwrap();
        let batches: Vec<_> = table_scan
            .to_arrow()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(total_rows, 1024);
        for batch in &batches {
            let x_col = batch.column_by_name("x").unwrap();
            // Plain declared scan-schema type, no error from the null partition value.
            assert_eq!(x_col.data_type(), &arrow_schema::DataType::Int64);
            let x = x_col.as_any().downcast_ref::<Int64Array>().unwrap();
            // No override (null is skipped): the file value is read.
            assert_eq!(x.value(0), 1);
            let y = batch.column_by_name("y").unwrap();
            let y = y.as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(y.value(0), 2);
        }
    }

    #[tokio::test]
    async fn test_filter_on_arrow_lt() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: y < 3
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("y").less_than(Datum::long(3));
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches[0].num_rows(), 512);

        let col = batches[0].column_by_name("x").unwrap();
        // `x` is an identity-partition constant (run-end-encoded); decode it.
        let int64_arr = decode_int64_column(col);
        assert_eq!(int64_arr.value(0), 1);

        let col = batches[0].column_by_name("y").unwrap();
        let int64_arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int64_arr.value(0), 2);
    }

    #[tokio::test]
    async fn test_filter_on_arrow_gt_eq() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: y >= 5
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("y").greater_than_or_equal_to(Datum::long(5));
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches[0].num_rows(), 12);

        let col = batches[0].column_by_name("x").unwrap();
        // `x` is an identity-partition constant (run-end-encoded); decode it.
        let int64_arr = decode_int64_column(col);
        assert_eq!(int64_arr.value(0), 1);

        let col = batches[0].column_by_name("y").unwrap();
        let int64_arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int64_arr.value(0), 5);
    }

    #[tokio::test]
    async fn test_filter_double_eq() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: dbl == 150.0
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("dbl").equal_to(Datum::double(150.0f64));
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 12);

        let col = batches[0].column_by_name("dbl").unwrap();
        let f64_arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(f64_arr.value(1), 150.0f64);
    }

    #[tokio::test]
    async fn test_filter_int_eq() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: i32 == 150
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("i32").equal_to(Datum::int(150i32));
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 12);

        let col = batches[0].column_by_name("i32").unwrap();
        let i32_arr = col.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(i32_arr.value(1), 150i32);
    }

    #[tokio::test]
    async fn test_filter_long_eq() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: i64 == 150
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("i64").equal_to(Datum::long(150i64));
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 12);

        let col = batches[0].column_by_name("i64").unwrap();
        let i64_arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(i64_arr.value(1), 150i64);
    }

    #[tokio::test]
    async fn test_filter_bool_eq() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: bool == true
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("bool").equal_to(Datum::bool(true));
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 512);

        let col = batches[0].column_by_name("bool").unwrap();
        let bool_arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(bool_arr.value(1));
    }

    #[tokio::test]
    async fn test_filter_on_arrow_is_null() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: y is null
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("y").is_null();
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();
        assert_eq!(batches.len(), 0);
    }

    #[tokio::test]
    async fn test_filter_on_arrow_is_not_null() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: y is not null
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("y").is_not_null();
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();
        assert_eq!(batches[0].num_rows(), 1024);
    }

    #[tokio::test]
    async fn test_filter_on_arrow_lt_and_gt() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: y < 5 AND z >= 4
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("y")
            .less_than(Datum::long(5))
            .and(Reference::new("z").greater_than_or_equal_to(Datum::long(4)));
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();
        assert_eq!(batches[0].num_rows(), 500);

        let col = batches[0].column_by_name("x").unwrap();
        // `x` is an identity-partition constant (run-end-encoded); decode it to the
        // logical flat array before comparing.
        let expected_x = Int64Array::from_iter_values(vec![1; 500]);
        assert_eq!(decode_int64_column(col), expected_x);

        let col = batches[0].column_by_name("y").unwrap();
        let mut values = vec![];
        values.append(vec![3; 200].as_mut());
        values.append(vec![4; 300].as_mut());
        let expected_y = Arc::new(Int64Array::from_iter_values(values)) as ArrayRef;
        assert_eq!(col, &expected_y);

        let col = batches[0].column_by_name("z").unwrap();
        let expected_z = Arc::new(Int64Array::from_iter_values(vec![4; 500])) as ArrayRef;
        assert_eq!(col, &expected_z);
    }

    #[tokio::test]
    async fn test_filter_on_arrow_lt_or_gt() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: y < 5 AND z >= 4
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("y")
            .less_than(Datum::long(5))
            .or(Reference::new("z").greater_than_or_equal_to(Datum::long(4)));
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();
        assert_eq!(batches[0].num_rows(), 1024);

        let col = batches[0].column_by_name("x").unwrap();
        // `x` is an identity-partition constant (run-end-encoded); decode it to the
        // logical flat array before comparing.
        let expected_x = Int64Array::from_iter_values(vec![1; 1024]);
        assert_eq!(decode_int64_column(col), expected_x);

        let col = batches[0].column_by_name("y").unwrap();
        let mut values = vec![2; 512];
        values.append(vec![3; 200].as_mut());
        values.append(vec![4; 300].as_mut());
        values.append(vec![5; 12].as_mut());
        let expected_y = Arc::new(Int64Array::from_iter_values(values)) as ArrayRef;
        assert_eq!(col, &expected_y);

        let col = batches[0].column_by_name("z").unwrap();
        let mut values = vec![3; 512];
        values.append(vec![4; 512].as_mut());
        let expected_z = Arc::new(Int64Array::from_iter_values(values)) as ArrayRef;
        assert_eq!(col, &expected_z);
    }

    // ---- Residual wiring (Phase 3, Increment 2) ----
    //
    // Each `FileScanTask` must carry the PARTITION-REDUCED residual of the scan
    // filter (Java `BaseFileScanTask.residual()` = `residuals.residualFor(
    // file.partition())`), not the full snapshot filter, and the file's partition
    // spec (resolving the old `partition_spec: None` TODO). These tests pin both,
    // plus result-equivalence: filtering rows with the residual yields the SAME
    // rows as the full filter would, because every row in a file shares the file's
    // single partition.
    //
    // The shared `TableTestFixture` is partitioned by `identity(x)` with partition
    // value `x == 1` on every live file (consistent with the parquet `x` column).

    /// Plans the files of a filtered scan over the shared fixture and returns the
    /// resulting tasks, sorted by data-file path for deterministic assertions.
    async fn plan_filtered_tasks(predicate: Predicate) -> Vec<FileScanTask> {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let table_scan = fixture.table.scan().with_filter(predicate).build().unwrap();

        let mut tasks: Vec<FileScanTask> = table_scan
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        tasks.sort_by_key(|task| task.data_file_path.to_string());
        tasks
    }

    #[tokio::test]
    async fn test_task_predicate_is_partition_reduced_residual_not_full_filter() {
        // Filter `x == 1 AND y > 0`: the `x == 1` leaf is fully implied by the
        // identity partition (x == 1) for every live file, so the residual must
        // drop it, leaving ONLY `y > 0`. The full filter would still carry the
        // `x == 1` leaf — leaving `predicate` as the full filter fails this.
        let filter = Reference::new("x")
            .equal_to(Datum::long(1))
            .and(Reference::new("y").greater_than(Datum::long(0)));
        let tasks = plan_filtered_tasks(filter).await;

        // Two live files (1.parquet, 3.parquet), both in partition x == 1.
        assert_eq!(tasks.len(), 2);

        let expected_residual = Reference::new("y")
            .greater_than(Datum::long(0))
            .bind(tasks[0].schema.clone(), true)
            .unwrap();
        for task in &tasks {
            assert_eq!(
                task.predicate.as_ref(),
                Some(&expected_residual),
                "task predicate must be the reduced residual `y > 0`, not the full \
                 `x == 1 AND y > 0` filter"
            );
        }
    }

    #[tokio::test]
    async fn test_task_predicate_residual_is_always_true_when_filter_fully_implied() {
        // Filter `x == 1` is ENTIRELY decided by the partition (x == 1) for every
        // live file → the residual is `AlwaysTrue` (no per-row filtering needed).
        let filter = Reference::new("x").equal_to(Datum::long(1));
        let tasks = plan_filtered_tasks(filter).await;

        assert_eq!(tasks.len(), 2);
        for task in &tasks {
            assert_eq!(
                task.predicate,
                Some(BoundPredicate::AlwaysTrue),
                "a filter fully implied by the partition must reduce to AlwaysTrue"
            );
        }
    }

    #[tokio::test]
    async fn test_residual_uses_files_own_spec_not_the_table_default_spec() {
        // RISK (silent scan-hot-path data bug): after partition evolution, an older
        // manifest's `partition_spec_id` differs from the table's DEFAULT spec. The
        // residual + threaded spec MUST come from the FILE's own spec
        // (`manifest_file.partition_spec_id`), NOT the table default. Here the live
        // file is written under spec 0 (`identity(x)`) while the table default is the
        // unpartitioned spec 1.
        //
        // With filter `x == 1`:
        //   - file's own spec (0, identity(x)) → `x == 1` is partition-implied →
        //     residual reduces to `AlwaysTrue`, and `task.partition_spec` is spec 0.
        //   - table default (1, unpartitioned) → NO reduction → residual stays the
        //     full `x == 1`, and the spec would be id 1.
        // Using the default spec instead of the file's own fails BOTH assertions.
        let mut fixture = TableTestFixture::new_with_evolved_default_spec();
        fixture.setup_manifest_files_under_spec_zero().await;

        let filter = Reference::new("x").equal_to(Datum::long(1));
        let table_scan = fixture.table.scan().with_filter(filter).build().unwrap();
        let tasks: Vec<FileScanTask> = table_scan
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        assert_eq!(tasks.len(), 1);
        let task = &tasks[0];

        // The residual was reduced by the FILE's identity(x) spec → AlwaysTrue. Using
        // the table default (unpartitioned) spec instead would leave the residual as the
        // full `x == 1` (no reduction), so this predicate assertion pins that the residual
        // is computed from the FILE's own `partition_spec_id`, not the table default.
        assert_eq!(
            task.predicate,
            Some(BoundPredicate::AlwaysTrue),
            "residual must be reduced by the file's own identity(x) spec (0), not \
             the unpartitioned default spec (1)"
        );
    }

    #[tokio::test]
    async fn test_residual_scan_result_equivalent_to_full_filter_identity_partition() {
        // RESULT-EQUIVALENCE on the identity-partitioned table. Filter mixes a
        // partition-implied leaf (`x == 1`) with a data-column leaf (`y < 3`). The
        // residual drops `x == 1` (always true for the x==1 files) and keeps
        // `y < 3`. The rows returned must EXACTLY equal the known-correct set the
        // full `x == 1 AND y < 3` filter selects: x == 1 holds for every row (the
        // partition constant), so the two filters are equivalent — `y < 3` selects
        // the 512 `y == 2` rows in EACH of the two live files (1.parquet, 3.parquet
        // carry identical data), 1024 rows total.
        let filter = Reference::new("x")
            .equal_to(Datum::long(1))
            .and(Reference::new("y").less_than(Datum::long(3)));

        // First confirm the per-task residual is genuinely the REDUCED form
        // (`y < 3`), so the read below exercises the residual path, not the full
        // filter.
        let tasks = plan_filtered_tasks(filter.clone()).await;
        assert_eq!(tasks.len(), 2);
        let expected_residual = Reference::new("y")
            .less_than(Datum::long(3))
            .bind(tasks[0].schema.clone(), true)
            .unwrap();
        assert_eq!(tasks[0].predicate.as_ref(), Some(&expected_residual));

        // Now read through the residual and assert the known-correct row set.
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;
        let table_scan = fixture
            .table
            .scan()
            .with_filter(filter)
            .with_row_selection_enabled(true)
            .build()
            .unwrap();
        let batches: Vec<_> = table_scan
            .to_arrow()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(
            total_rows, 1024,
            "y < 3 selects the 512 y == 2 rows in each of the 2 live files"
        );
        for batch in &batches {
            // Every selected row has x == 1 (the partition value) and y == 2.
            let x = decode_int64_column(batch.column_by_name("x").unwrap());
            for index in 0..x.len() {
                assert_eq!(x.value(index), 1);
            }
            let y = batch.column_by_name("y").unwrap();
            let y = y.as_any().downcast_ref::<Int64Array>().unwrap();
            for index in 0..y.len() {
                assert_eq!(y.value(index), 2);
            }
        }
    }

    #[tokio::test]
    async fn test_unpartitioned_table_keeps_full_filter_as_task_predicate() {
        // On an unpartitioned table the residual equals the full filter — no
        // reduction — so `task.predicate` must be the full bound filter unchanged.
        let mut fixture = TableTestFixture::new_unpartitioned();
        fixture.setup_unpartitioned_manifest_files().await;

        let filter = Reference::new("y")
            .greater_than(Datum::long(0))
            .and(Reference::new("z").less_than(Datum::long(5)));
        let table_scan = fixture
            .table
            .scan()
            .with_filter(filter.clone())
            .build()
            .unwrap();
        let tasks: Vec<FileScanTask> = table_scan
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        assert!(!tasks.is_empty());
        let expected = filter.bind(tasks[0].schema.clone(), true).unwrap();
        for task in &tasks {
            assert_eq!(
                task.predicate.as_ref(),
                Some(&expected),
                "an unpartitioned table's residual is the full filter, unchanged"
            );
        }
    }

    #[tokio::test]
    async fn test_no_filter_scan_leaves_task_predicate_none() {
        // A scan with no row filter has no residual evaluator → no per-task
        // predicate (behavior unchanged from before the wiring).
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let table_scan = fixture.table.scan().build().unwrap();
        let tasks: Vec<FileScanTask> = table_scan
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        assert_eq!(tasks.len(), 2);
        for task in &tasks {
            assert_eq!(task.predicate, None);
        }
    }

    #[tokio::test]
    async fn test_filter_excluding_the_partition_prunes_the_file_entirely() {
        // Filter `x == 999`: no live file is in partition x == 999, so the
        // inclusive projection is false for every file and they are all pruned
        // (manifest + expression evaluators) — zero tasks. This is the partition
        // mismatch the residual's AlwaysFalse case corresponds to at planning time.
        let filter = Reference::new("x").equal_to(Datum::long(999));
        let tasks = plan_filtered_tasks(filter).await;
        assert!(
            tasks.is_empty(),
            "no file is in partition x == 999, so all are pruned"
        );
    }

    #[tokio::test]
    async fn test_task_predicate_residual_reduced_on_transform_truncate_partition() {
        // TRANSFORM partition (`truncate(x, 100)`), file in partition bucket 0
        // (x in 0..=99). Filter `x < 100 AND y > 0`: the truncate-0 partition's
        // STRICT projection of `x < 100` is true (every x in 0..=99 is < 100), so
        // the residual drops the `x < 100` leaf, leaving only `y > 0` — proving the
        // wiring reduces a NON-identity partition predicate, not just identity.
        let mut fixture = TableTestFixture::new_truncate_partitioned();
        fixture.setup_truncate_manifest_files().await;

        let filter = Reference::new("x")
            .less_than(Datum::long(100))
            .and(Reference::new("y").greater_than(Datum::long(0)));
        let table_scan = fixture.table.scan().with_filter(filter).build().unwrap();
        let tasks: Vec<FileScanTask> = table_scan
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        assert_eq!(tasks.len(), 1);
        let expected_residual = Reference::new("y")
            .greater_than(Datum::long(0))
            .bind(tasks[0].schema.clone(), true)
            .unwrap();
        assert_eq!(
            tasks[0].predicate.as_ref(),
            Some(&expected_residual),
            "truncate(x,100)==0 strictly implies x < 100, so the residual is `y > 0`"
        );
    }

    #[tokio::test]
    async fn test_residual_scan_result_equivalent_to_full_filter_truncate_partition() {
        // RESULT-EQUIVALENCE on the truncate-partitioned table. Filter
        // `x < 100 AND y >= 4`: truncate(x,100)==0 implies x < 100 for every row, so
        // the residual `y >= 4` selects the SAME rows as the full filter. The data
        // has y >= 4 for 312 rows (300 of y==4 + 12 of y==5). Because the truncate
        // transform is NON-identity, `x` is read from the parquet file (= 1), so the
        // full-filter leaf `x < 100` is also true for every row — equivalence holds.
        let mut fixture = TableTestFixture::new_truncate_partitioned();
        fixture.setup_truncate_manifest_files().await;

        let filter = Reference::new("x")
            .less_than(Datum::long(100))
            .and(Reference::new("y").greater_than_or_equal_to(Datum::long(4)));
        let table_scan = fixture
            .table
            .scan()
            .with_filter(filter)
            .with_row_selection_enabled(true)
            .build()
            .unwrap();
        let batches: Vec<_> = table_scan
            .to_arrow()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(
            total_rows, 312,
            "y >= 4 selects 300 (y==4) + 12 (y==5) rows"
        );
        for batch in &batches {
            // x is read from the data file (1) — NOT a constant, since truncate is
            // non-identity — and every selected row satisfies x < 100 AND y >= 4.
            let x = decode_int64_column(batch.column_by_name("x").unwrap());
            for index in 0..x.len() {
                assert_eq!(x.value(index), 1);
            }
            let y = batch.column_by_name("y").unwrap();
            let y = y.as_any().downcast_ref::<Int64Array>().unwrap();
            for index in 0..y.len() {
                assert!(y.value(index) >= 4);
            }
        }
    }

    #[tokio::test]
    async fn test_filter_on_arrow_startswith() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: a STARTSWITH "Ice"
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("a").starts_with(Datum::string("Ice"));
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches[0].num_rows(), 512);

        let col = batches[0].column_by_name("a").unwrap();
        let string_arr = col.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(string_arr.value(0), "Iceberg");
    }

    #[tokio::test]
    async fn test_filter_on_arrow_not_startswith() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: a NOT STARTSWITH "Ice"
        let mut builder = fixture.table.scan();
        let predicate = Reference::new("a").not_starts_with(Datum::string("Ice"));
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches[0].num_rows(), 512);

        let col = batches[0].column_by_name("a").unwrap();
        let string_arr = col.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(string_arr.value(0), "Apache");
    }

    #[tokio::test]
    async fn test_filter_on_arrow_in() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: a IN ("Sioux", "Iceberg")
        let mut builder = fixture.table.scan();
        let predicate =
            Reference::new("a").is_in([Datum::string("Sioux"), Datum::string("Iceberg")]);
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches[0].num_rows(), 512);

        let col = batches[0].column_by_name("a").unwrap();
        let string_arr = col.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(string_arr.value(0), "Iceberg");
    }

    #[tokio::test]
    async fn test_filter_on_arrow_not_in() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Filter: a NOT IN ("Sioux", "Iceberg")
        let mut builder = fixture.table.scan();
        let predicate =
            Reference::new("a").is_not_in([Datum::string("Sioux"), Datum::string("Iceberg")]);
        builder = builder
            .with_filter(predicate)
            .with_row_selection_enabled(true);
        let table_scan = builder.build().unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();

        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches[0].num_rows(), 512);

        let col = batches[0].column_by_name("a").unwrap();
        let string_arr = col.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(string_arr.value(0), "Apache");
    }

    #[test]
    fn test_file_scan_task_serialize_deserialize() {
        let test_fn = |task: FileScanTask| {
            let serialized = serde_json::to_string(&task).unwrap();
            let deserialized: FileScanTask = serde_json::from_str(&serialized).unwrap();

            assert_eq!(task.data_file_path, deserialized.data_file_path);
            assert_eq!(task.start, deserialized.start);
            assert_eq!(task.length, deserialized.length);
            assert_eq!(task.project_field_ids, deserialized.project_field_ids);
            assert_eq!(task.predicate, deserialized.predicate);
            assert_eq!(task.schema, deserialized.schema);
        };

        // without predicate
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![Arc::new(NestedField::required(
                    1,
                    "x",
                    Type::Primitive(PrimitiveType::Binary),
                ))])
                .build()
                .unwrap(),
        );
        let task = FileScanTask {
            data_file_path: "data_file_path".to_string(),
            file_size_in_bytes: 0,
            start: 0,
            length: 100,
            project_field_ids: vec![1, 2, 3],
            predicate: None,
            schema: schema.clone(),
            record_count: Some(100),
            data_file_format: DataFileFormat::Parquet,
            deletes: vec![],
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
        };
        test_fn(task);

        // with predicate
        let task = FileScanTask {
            data_file_path: "data_file_path".to_string(),
            file_size_in_bytes: 0,
            start: 0,
            length: 100,
            project_field_ids: vec![1, 2, 3],
            predicate: Some(BoundPredicate::AlwaysTrue),
            schema,
            record_count: None,
            data_file_format: DataFileFormat::Avro,
            deletes: vec![],
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
        };
        test_fn(task);
    }

    #[tokio::test]
    async fn test_select_with_file_column() {
        use arrow_array::cast::AsArray;

        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Select regular columns plus the _file column
        let table_scan = fixture
            .table
            .scan()
            .select(["x", RESERVED_COL_NAME_FILE])
            .with_row_selection_enabled(true)
            .build()
            .unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();
        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        // Verify we have 2 columns: x and _file
        assert_eq!(batches[0].num_columns(), 2);

        // Verify the x column exists and has correct data. `x` is an
        // identity-partition constant (run-end-encoded); decode it.
        let x_col = batches[0].column_by_name("x").unwrap();
        let x_arr = decode_int64_column(x_col);
        assert_eq!(x_arr.value(0), 1);

        // Verify the _file column exists
        let file_col = batches[0].column_by_name(RESERVED_COL_NAME_FILE);
        assert!(
            file_col.is_some(),
            "_file column should be present in the batch"
        );

        // Verify the _file column contains a file path
        let file_col = file_col.unwrap();
        assert!(
            matches!(
                file_col.data_type(),
                arrow_schema::DataType::RunEndEncoded(_, _)
            ),
            "_file column should use RunEndEncoded type"
        );

        // Decode the RunArray to verify it contains the file path
        let run_array = file_col
            .as_any()
            .downcast_ref::<arrow_array::RunArray<arrow_array::types::Int32Type>>()
            .expect("_file column should be a RunArray");

        let values = run_array.values();
        let string_values = values.as_string::<i32>();
        assert_eq!(string_values.len(), 1, "Should have a single file path");

        let file_path = string_values.value(0);
        assert!(
            file_path.ends_with(".parquet"),
            "File path should end with .parquet, got: {file_path}"
        );
    }

    #[tokio::test]
    async fn test_select_file_column_position() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Select columns in specific order: x, _file, z
        let table_scan = fixture
            .table
            .scan()
            .select(["x", RESERVED_COL_NAME_FILE, "z"])
            .with_row_selection_enabled(true)
            .build()
            .unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();
        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches[0].num_columns(), 3);

        // Verify column order: x at position 0, _file at position 1, z at position 2
        let schema = batches[0].schema();
        assert_eq!(schema.field(0).name(), "x");
        assert_eq!(schema.field(1).name(), RESERVED_COL_NAME_FILE);
        assert_eq!(schema.field(2).name(), "z");

        // Verify columns by name also works
        assert!(batches[0].column_by_name("x").is_some());
        assert!(batches[0].column_by_name(RESERVED_COL_NAME_FILE).is_some());
        assert!(batches[0].column_by_name("z").is_some());
    }

    #[tokio::test]
    async fn test_select_file_column_only() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Select only the _file column
        let table_scan = fixture
            .table
            .scan()
            .select([RESERVED_COL_NAME_FILE])
            .with_row_selection_enabled(true)
            .build()
            .unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();
        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        // Should have exactly 1 column
        assert_eq!(batches[0].num_columns(), 1);

        // Verify it's the _file column
        let schema = batches[0].schema();
        assert_eq!(schema.field(0).name(), RESERVED_COL_NAME_FILE);

        // Verify the batch has the correct number of rows
        // The scan reads files 1.parquet and 3.parquet (2.parquet is deleted)
        // Each file has 1024 rows, so total is 2048 rows
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2048);
    }

    #[tokio::test]
    async fn test_file_column_with_multiple_files() {
        use std::collections::HashSet;

        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Select x and _file columns
        let table_scan = fixture
            .table
            .scan()
            .select(["x", RESERVED_COL_NAME_FILE])
            .with_row_selection_enabled(true)
            .build()
            .unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();
        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        // Collect all unique file paths from the batches
        let mut file_paths = HashSet::new();
        for batch in &batches {
            let file_col = batch.column_by_name(RESERVED_COL_NAME_FILE).unwrap();
            let run_array = file_col
                .as_any()
                .downcast_ref::<arrow_array::RunArray<arrow_array::types::Int32Type>>()
                .expect("_file column should be a RunArray");

            let values = run_array.values();
            let string_values = values.as_string::<i32>();
            for i in 0..string_values.len() {
                file_paths.insert(string_values.value(i).to_string());
            }
        }

        // We should have multiple files (the test creates 1.parquet and 3.parquet)
        assert!(!file_paths.is_empty(), "Should have at least one file path");

        // All paths should end with .parquet
        for path in &file_paths {
            assert!(
                path.ends_with(".parquet"),
                "All file paths should end with .parquet, got: {path}"
            );
        }
    }

    #[tokio::test]
    async fn test_file_column_at_start() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Select _file at the start
        let table_scan = fixture
            .table
            .scan()
            .select([RESERVED_COL_NAME_FILE, "x", "y"])
            .with_row_selection_enabled(true)
            .build()
            .unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();
        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches[0].num_columns(), 3);

        // Verify _file is at position 0
        let schema = batches[0].schema();
        assert_eq!(schema.field(0).name(), RESERVED_COL_NAME_FILE);
        assert_eq!(schema.field(1).name(), "x");
        assert_eq!(schema.field(2).name(), "y");
    }

    #[tokio::test]
    async fn test_file_column_at_end() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Select _file at the end
        let table_scan = fixture
            .table
            .scan()
            .select(["x", "y", RESERVED_COL_NAME_FILE])
            .with_row_selection_enabled(true)
            .build()
            .unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();
        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        assert_eq!(batches[0].num_columns(), 3);

        // Verify _file is at position 2 (the end)
        let schema = batches[0].schema();
        assert_eq!(schema.field(0).name(), "x");
        assert_eq!(schema.field(1).name(), "y");
        assert_eq!(schema.field(2).name(), RESERVED_COL_NAME_FILE);
    }

    #[tokio::test]
    async fn test_select_with_repeated_column_names() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Select with repeated column names - both regular columns and virtual columns
        // Repeated columns should appear multiple times in the result (duplicates are allowed)
        let table_scan = fixture
            .table
            .scan()
            .select([
                "x",
                RESERVED_COL_NAME_FILE,
                "x", // x repeated
                "y",
                RESERVED_COL_NAME_FILE, // _file repeated
                "y",                    // y repeated
            ])
            .with_row_selection_enabled(true)
            .build()
            .unwrap();

        let batch_stream = table_scan.to_arrow().await.unwrap();
        let batches: Vec<_> = batch_stream.try_collect().await.unwrap();

        // Verify we have exactly 6 columns (duplicates are allowed and preserved)
        assert_eq!(
            batches[0].num_columns(),
            6,
            "Should have exactly 6 columns with duplicates"
        );

        let schema = batches[0].schema();

        // Verify columns appear in the exact order requested: x, _file, x, y, _file, y
        assert_eq!(schema.field(0).name(), "x", "Column 0 should be x");
        assert_eq!(
            schema.field(1).name(),
            RESERVED_COL_NAME_FILE,
            "Column 1 should be _file"
        );
        assert_eq!(
            schema.field(2).name(),
            "x",
            "Column 2 should be x (duplicate)"
        );
        assert_eq!(schema.field(3).name(), "y", "Column 3 should be y");
        assert_eq!(
            schema.field(4).name(),
            RESERVED_COL_NAME_FILE,
            "Column 4 should be _file (duplicate)"
        );
        assert_eq!(
            schema.field(5).name(),
            "y",
            "Column 5 should be y (duplicate)"
        );

        // Verify all columns have correct data types. `x` is read from the data file as
        // a plain `Int64` (identity-partition constant materialization is deferred — see
        // `into_file_scan_task`, where `partition_spec` is left `None`).
        assert!(
            matches!(schema.field(0).data_type(), arrow_schema::DataType::Int64),
            "Column x should be a plain Int64 read from the data file"
        );
        assert!(
            matches!(schema.field(2).data_type(), arrow_schema::DataType::Int64),
            "Column x (duplicate) should be a plain Int64 read from the data file"
        );
        assert!(
            matches!(schema.field(3).data_type(), arrow_schema::DataType::Int64),
            "Column y should be Int64"
        );
        assert!(
            matches!(schema.field(5).data_type(), arrow_schema::DataType::Int64),
            "Column y (duplicate) should be Int64"
        );
        assert!(
            matches!(
                schema.field(1).data_type(),
                arrow_schema::DataType::RunEndEncoded(_, _)
            ),
            "_file column should use RunEndEncoded type"
        );
        assert!(
            matches!(
                schema.field(4).data_type(),
                arrow_schema::DataType::RunEndEncoded(_, _)
            ),
            "_file column (duplicate) should use RunEndEncoded type"
        );
    }

    #[tokio::test]
    async fn test_scan_deadlock() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_deadlock_manifests().await;

        // Create table scan with concurrency limit 1
        // This sets channel size to 1.
        // Data manifest has 10 entries -> will block producer.
        // Delete manifest is 2nd in list -> won't be processed.
        // Consumer 2 (Data) not started -> blocked.
        // Consumer 1 (Delete) waiting -> blocked.
        let table_scan = fixture
            .table
            .scan()
            .with_concurrency_limit(1)
            .build()
            .unwrap();

        // This should timeout/hang if deadlock exists
        // We can use tokio::time::timeout
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            table_scan
                .plan_files()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
        })
        .await;

        // Assert it finished (didn't timeout)
        assert!(result.is_ok(), "Scan timed out - deadlock detected");
    }

    // =======================================================================================
    // BatchScan (Java BatchScan / BatchScanAdapter) — the typed adapter over the SAME pipeline.
    //
    // The contract: `table.batch_scan()` is a thin typed wrapper whose plan_files()/plan_tasks()
    // DELEGATE 1:1 to the TableScan pipeline (Java BatchScanAdapter), plus the three time-travel
    // selectors use_snapshot / use_ref / as_of_time (Java SnapshotScan.useSnapshot/useRef/asOfTime,
    // first-wins mutual exclusion). The fixture's snapshot LOG is
    //   {id 3051729675574597004 @ 1515100955770} then {id 3055729675574597004 @ 1555100955770}
    // with current = 3055729675574597004 and a `test` tag → 3051729675574597004.
    // =======================================================================================

    /// Older snapshot-log entry's timestamp (id 3051729675574597004).
    const SNAP1_TS: i64 = 1515100955770;
    /// Newer snapshot-log entry's timestamp (id 3055729675574597004 == current).
    const SNAP2_TS: i64 = 1555100955770;

    /// The canonical group multiset for a list of `CombinedScanTask`s: each group is the SORTED set
    /// of member keys `(path,start,length)`, then the list of groups is sorted (order-insensitive
    /// across groups, duplicate-preserving) — the same comparison contract as the interop oracle.
    fn group_multiset(groups: &[CombinedScanTask]) -> Vec<Vec<String>> {
        let mut out: Vec<Vec<String>> = groups
            .iter()
            .map(|group| {
                let mut members: Vec<String> = group
                    .tasks()
                    .iter()
                    .map(|task| {
                        format!("({},{},{})", task.data_file_path(), task.start, task.length)
                    })
                    .collect();
                members.sort();
                members
            })
            .collect();
        out.sort();
        out
    }

    /// DELEGATION (the crux): `BatchScan::plan_tasks()` must yield the SAME groups as
    /// `TableScan::plan_tasks()` over the same table + the same split knobs. This is the Java
    /// `table.newBatchScan().planTasks() == table.newScan().planTasks()` adapter identity.
    #[tokio::test]
    async fn test_batch_scan_plan_tasks_delegates_to_table_scan() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_for_planning().await;

        // A non-trivial split + bin-pack config so the delegation covers real grouping, not a
        // degenerate single group.
        let table_scan = fixture
            .table
            .scan()
            .with_split_size(500)
            .with_split_lookback(10)
            .with_split_open_file_cost(0)
            .build()
            .unwrap();
        let scan_groups: Vec<CombinedScanTask> = table_scan
            .plan_tasks()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let batch_groups: Vec<CombinedScanTask> = fixture
            .table
            .batch_scan()
            .with_split_size(500)
            .with_split_lookback(10)
            .with_split_open_file_cost(0)
            .plan_tasks()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        assert_eq!(
            group_multiset(&batch_groups),
            group_multiset(&scan_groups),
            "BatchScan::plan_tasks must yield the SAME groups as TableScan::plan_tasks (adapter delegation)"
        );
        // Sanity: the fixture actually produces a multi-group plan (otherwise the delegation is
        // vacuously trivial). 7 sub-tasks at target 500 cannot fit in one bin.
        assert!(
            scan_groups.len() > 1,
            "the delegation fixture must produce >1 group (got {})",
            scan_groups.len()
        );
    }

    /// DELEGATION for `plan_files()`: the BatchScan file-scan-task stream must match the TableScan
    /// one (same member keys), since the adapter forwards `planFiles()` 1:1.
    #[tokio::test]
    async fn test_batch_scan_plan_files_delegates_to_table_scan() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_for_planning().await;

        let scan_keys: BTreeSet<String> = fixture
            .table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .iter()
            .map(|task| format!("({},{},{})", task.data_file_path(), task.start, task.length))
            .collect();

        let batch_keys: BTreeSet<String> = fixture
            .table
            .batch_scan()
            .plan_files()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .iter()
            .map(|task| format!("({},{},{})", task.data_file_path(), task.start, task.length))
            .collect();

        assert_eq!(
            batch_keys, scan_keys,
            "BatchScan::plan_files must match TableScan::plan_files (adapter delegation)"
        );
        assert!(
            !scan_keys.is_empty(),
            "the fixture must yield file scan tasks"
        );
    }

    /// `batch_scan()` with NO selector reads the table default (current snapshot), like a plain
    /// `scan()` and like Java's `useRef("main")` baseline.
    #[tokio::test]
    async fn test_batch_scan_default_is_current_snapshot() {
        let table = TableTestFixture::new().table;
        let snapshot = table
            .batch_scan()
            .snapshot()
            .unwrap()
            .expect("a current snapshot");
        assert_eq!(snapshot.snapshot_id(), CURRENT_SNAPSHOT_ID);
    }

    /// `use_snapshot(id)` pins THAT snapshot (Java `BatchScan.useSnapshot`).
    #[tokio::test]
    async fn test_batch_scan_use_snapshot_selects_that_snapshot() {
        let table = TableTestFixture::new().table;
        let snapshot = table
            .batch_scan()
            .use_snapshot(TAG_TEST_SNAPSHOT_ID)
            .snapshot()
            .unwrap()
            .expect("the pinned snapshot");
        assert_eq!(snapshot.snapshot_id(), TAG_TEST_SNAPSHOT_ID);
        assert_ne!(snapshot.snapshot_id(), CURRENT_SNAPSHOT_ID);
    }

    /// `use_ref("main")` resolves to the table default; `use_ref("test")` resolves to the tagged
    /// (older) snapshot — NOT the current one (Java `BatchScan.useRef`).
    #[tokio::test]
    async fn test_batch_scan_use_ref() {
        let table = TableTestFixture::new().table;

        let main = table
            .batch_scan()
            .use_ref("main")
            .snapshot()
            .unwrap()
            .expect("main snapshot");
        assert_eq!(main.snapshot_id(), CURRENT_SNAPSHOT_ID);

        let tagged = table
            .batch_scan()
            .use_ref("test")
            .snapshot()
            .unwrap()
            .expect("tag snapshot");
        assert_eq!(tagged.snapshot_id(), TAG_TEST_SNAPSHOT_ID);
        assert_ne!(tagged.snapshot_id(), CURRENT_SNAPSHOT_ID);
    }

    /// An unknown ref surfaces as a typed error (Java "Cannot find ref %s"), naming the ref.
    #[tokio::test]
    async fn test_batch_scan_use_ref_unknown_errors() {
        let table = TableTestFixture::new().table;
        let error = table
            .batch_scan()
            .use_ref("does_not_exist")
            .snapshot()
            .expect_err("unknown ref must error");
        assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
        assert!(
            error.message().contains("does_not_exist"),
            "error should name the missing ref, got: {error}"
        );
    }

    /// `as_of_time` BOUNDARY — the inclusive `<=` over the snapshot LOG (Java
    /// `SnapshotUtil.snapshotIdAsOfTime`):
    ///   * EXACTLY at snap2's timestamp ⇒ snap2 (== current). MUTATION BAIT: a `<` boundary would
    ///     wrongly pick snap1; a "pick the earliest <=" bug would also pick snap1.
    ///   * 1ms BEFORE snap2 ⇒ snap1 (the greatest `<=` is snap1).
    ///   * EXACTLY at snap1's timestamp ⇒ snap1.
    #[tokio::test]
    async fn test_batch_scan_as_of_time_boundary_is_inclusive() {
        let table = TableTestFixture::new().table;

        // Exactly on snap2's timestamp ⇒ snap2 (the inclusive-boundary case; `<` would red this).
        let at_snap2 = table
            .batch_scan()
            .as_of_time(SNAP2_TS)
            .snapshot()
            .unwrap()
            .expect("a snapshot as of snap2's timestamp");
        assert_eq!(
            at_snap2.snapshot_id(),
            CURRENT_SNAPSHOT_ID,
            "as_of_time exactly at snap2's timestamp must select snap2 (inclusive <=)"
        );

        // 1ms before snap2 ⇒ snap1 (the greatest timestamp <= the arg is snap1).
        let just_before_snap2 = table
            .batch_scan()
            .as_of_time(SNAP2_TS - 1)
            .snapshot()
            .unwrap()
            .expect("a snapshot 1ms before snap2");
        assert_eq!(
            just_before_snap2.snapshot_id(),
            TAG_TEST_SNAPSHOT_ID,
            "as_of_time 1ms before snap2 must select snap1, not snap2"
        );

        // Exactly on snap1's timestamp ⇒ snap1 (inclusive again).
        let at_snap1 = table
            .batch_scan()
            .as_of_time(SNAP1_TS)
            .snapshot()
            .unwrap()
            .expect("a snapshot as of snap1's timestamp");
        assert_eq!(at_snap1.snapshot_id(), TAG_TEST_SNAPSHOT_ID);

        // Well after snap2 ⇒ still snap2 (the greatest <= is the latest entry).
        let after_all = table
            .batch_scan()
            .as_of_time(SNAP2_TS + 1_000_000)
            .snapshot()
            .unwrap()
            .expect("a snapshot well after snap2");
        assert_eq!(after_all.snapshot_id(), CURRENT_SNAPSHOT_ID);
    }

    /// `as_of_time` with NO log entry `<=` the timestamp errors (Java "Cannot find a snapshot older
    /// than %s"). MUTATION BAIT: a "pick the earliest snapshot regardless" bug would return snap1
    /// here instead of erroring.
    #[tokio::test]
    async fn test_batch_scan_as_of_time_before_all_errors() {
        let table = TableTestFixture::new().table;
        let before = SNAP1_TS - 1;
        let error = table
            .batch_scan()
            .as_of_time(before)
            .snapshot()
            .expect_err("a timestamp before every snapshot must error");
        assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .contains("Cannot find a snapshot older than"),
            "error should be the Java 'Cannot find a snapshot older than' message, got: {error}"
        );
        assert!(
            error.message().contains(&before.to_string()),
            "error should name the timestamp, got: {error}"
        );
    }

    /// `as_of_time` then `plan_tasks` plans the SELECTED snapshot — the time-travel selector flows
    /// through the delegated pipeline, not just `snapshot()`.
    #[tokio::test]
    async fn test_batch_scan_as_of_time_plans_selected_snapshot() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_for_planning().await;

        // as_of_time at snap2's timestamp == the current snapshot, which is the one the planning
        // manifest was written under ⇒ a non-empty plan equal to the default plan.
        let default_groups: Vec<CombinedScanTask> = fixture
            .table
            .batch_scan()
            .plan_tasks()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let as_of_groups: Vec<CombinedScanTask> = fixture
            .table
            .batch_scan()
            .as_of_time(SNAP2_TS)
            .plan_tasks()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert!(
            !default_groups.is_empty(),
            "the default plan must be non-empty"
        );
        assert_eq!(
            group_multiset(&as_of_groups),
            group_multiset(&default_groups),
            "as_of_time(current ts) must plan the same snapshot as the default"
        );
    }

    /// MUTUAL EXCLUSION — two snapshot-pinning selectors conflict FIRST-WINS (Java `checkArgument`
    /// "Cannot override snapshot/ref, already set snapshot id=%s"). The conflict is latched at the
    /// SECOND selector and surfaces as a typed error at plan/snapshot time.
    #[tokio::test]
    async fn test_batch_scan_conflicting_selectors_error() {
        let table = TableTestFixture::new().table;

        // use_snapshot then use_snapshot.
        let error = table
            .batch_scan()
            .use_snapshot(CURRENT_SNAPSHOT_ID)
            .use_snapshot(TAG_TEST_SNAPSHOT_ID)
            .snapshot()
            .expect_err("two use_snapshot calls must conflict");
        assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
        assert!(
            error.message().contains("Cannot override snapshot"),
            "got: {error}"
        );

        // use_snapshot then use_ref.
        let error = table
            .batch_scan()
            .use_snapshot(CURRENT_SNAPSHOT_ID)
            .use_ref("test")
            .snapshot()
            .expect_err("use_snapshot then use_ref must conflict");
        assert!(
            error.message().contains("Cannot override ref"),
            "got: {error}"
        );

        // use_snapshot then as_of_time, surfaced via the DELEGATED plan_tasks path (the stream type
        // is not Debug, so match the error out by hand rather than via expect_err).
        let result = table
            .batch_scan()
            .use_snapshot(CURRENT_SNAPSHOT_ID)
            .as_of_time(SNAP2_TS)
            .plan_tasks()
            .await;
        let error = match result {
            Ok(_) => panic!("use_snapshot then as_of_time must conflict"),
            Err(error) => error,
        };
        assert!(
            error.message().contains("Cannot override snapshot"),
            "got: {error}"
        );

        // as_of_time then use_snapshot (asOfTime pins via useSnapshot, so the SECOND conflicts).
        let error = table
            .batch_scan()
            .as_of_time(SNAP2_TS)
            .use_snapshot(TAG_TEST_SNAPSHOT_ID)
            .snapshot()
            .expect_err("as_of_time then use_snapshot must conflict");
        assert!(
            error.message().contains("Cannot override snapshot"),
            "got: {error}"
        );
    }

    /// `use_ref("main")` is the table-default no-op even AFTER another selector pinned a snapshot —
    /// it must NOT conflict (Java `useRef(MAIN)` early-returns without touching the pin). It also
    /// must NOT override the earlier pin.
    #[tokio::test]
    async fn test_batch_scan_use_ref_main_is_noop_not_conflict() {
        let table = TableTestFixture::new().table;
        let snapshot = table
            .batch_scan()
            .use_snapshot(TAG_TEST_SNAPSHOT_ID)
            .use_ref("main")
            .snapshot()
            .unwrap()
            .expect("the pinned snapshot survives a main no-op");
        assert_eq!(
            snapshot.snapshot_id(),
            TAG_TEST_SNAPSHOT_ID,
            "use_ref(main) must neither conflict nor override the earlier pin"
        );
    }

    // ===== Event listeners: a REAL scan genuinely fires a `ScanEvent` =====

    /// A `ScanEvent` listener that records every event into a shared sink.
    struct ScanEventRecorder {
        sink: Arc<std::sync::Mutex<Vec<crate::events::ScanEvent>>>,
    }

    impl crate::events::Listener<crate::events::ScanEvent> for ScanEventRecorder {
        fn notify(&self, event: &crate::events::ScanEvent) {
            self.sink.lock().unwrap().push(event.clone());
        }
    }

    /// Risk: the `ScanEvent` emit is wired but never actually fires on a real plan (a registry
    /// nothing fires is only a partial port). Pins that running `plan_files` over a REAL fixture
    /// table fires exactly one `ScanEvent` carrying the scanned snapshot id, the UNBOUND filter,
    /// the table name, and the projection schema id.
    #[tokio::test]
    async fn test_real_scan_fires_scan_event() {
        let _guard = crate::events::test_support::lock();

        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let sink = Arc::new(std::sync::Mutex::new(Vec::new()));
        crate::events::register::<crate::events::ScanEvent>(Arc::new(ScanEventRecorder {
            sink: sink.clone(),
        }));

        let predicate = Reference::new("y").greater_than(Datum::long(3));
        let scan = fixture
            .table
            .scan()
            .with_filter(predicate.clone())
            .select(["x", "y"])
            .build()
            .unwrap();
        let _tasks: Vec<FileScanTask> = scan
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let events = sink.lock().unwrap();
        assert_eq!(events.len(), 1, "a real scan fires exactly one ScanEvent");
        let event = &events[0];
        assert_eq!(event.snapshot_id(), CURRENT_SNAPSHOT_ID);
        assert_eq!(event.table_name(), "db.table1");
        // The filter is the UNBOUND row filter (after the builder's `rewrite_not` normalization),
        // NOT the snapshot-bound predicate.
        assert_eq!(event.filter(), &predicate);
        assert_eq!(
            event.projection().schema_id(),
            fixture
                .table
                .metadata()
                .current_snapshot()
                .unwrap()
                .schema(fixture.table.metadata())
                .unwrap()
                .schema_id()
        );
    }

    /// Risk: a scan with NO filter fires an event whose filter is something other than the
    /// `AlwaysTrue` default (Java `BaseScan.filter()` defaults to `alwaysTrue`). Pins the default.
    #[tokio::test]
    async fn test_real_scan_no_filter_fires_always_true() {
        let _guard = crate::events::test_support::lock();

        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let sink = Arc::new(std::sync::Mutex::new(Vec::new()));
        crate::events::register::<crate::events::ScanEvent>(Arc::new(ScanEventRecorder {
            sink: sink.clone(),
        }));

        let scan = fixture.table.scan().build().unwrap();
        let _tasks: Vec<FileScanTask> = scan
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let events = sink.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].filter(), &Predicate::AlwaysTrue);
    }

    /// Risk: an EMPTY (snapshotless) scan still fires a `ScanEvent`, diverging from Java
    /// `SnapshotScan.planFiles`, which returns empty BEFORE firing when there is no snapshot.
    /// Pins that a snapshotless scan fires NOTHING.
    #[tokio::test]
    async fn test_empty_scan_fires_no_event() {
        let _guard = crate::events::test_support::lock();

        let fixture = TableTestFixture::new_empty();

        let sink = Arc::new(std::sync::Mutex::new(Vec::new()));
        crate::events::register::<crate::events::ScanEvent>(Arc::new(ScanEventRecorder {
            sink: sink.clone(),
        }));

        let scan = fixture.table.scan().build().unwrap();
        let tasks: Vec<FileScanTask> = scan
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert!(tasks.is_empty(), "snapshotless scan yields no tasks");

        assert!(
            sink.lock().unwrap().is_empty(),
            "a snapshotless scan must fire no ScanEvent"
        );
    }

    /// Risk: the scan emit silently swallows a panicking listener (Java `SnapshotScan.planFiles`
    /// has NO guard — a throwing listener propagates). Pins that a panicking scan listener
    /// propagates out of `plan_files`.
    #[tokio::test]
    async fn test_scan_listener_panic_propagates() {
        let _guard = crate::events::test_support::lock();

        struct Panicker;
        impl crate::events::Listener<crate::events::ScanEvent> for Panicker {
            fn notify(&self, _event: &crate::events::ScanEvent) {
                panic!("scan listener boom");
            }
        }

        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;
        crate::events::register::<crate::events::ScanEvent>(Arc::new(Panicker));

        let scan = fixture.table.scan().build().unwrap();
        let result = std::panic::AssertUnwindSafe(async {
            let _ = scan.plan_files().await;
        });
        let outcome = futures::FutureExt::catch_unwind(result).await;
        assert!(
            outcome.is_err(),
            "a panicking scan listener must propagate out of plan_files (Java has no guard)"
        );
    }
}
