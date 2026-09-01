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
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, RecordBatch, StringArray, UInt64Array};
use datafusion::arrow::datatypes::{
    DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::StreamExt;
use iceberg::Catalog;
use iceberg::expr::Predicate;
use iceberg::spec::{DataFile, deserialize_data_file_from_json};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};

use crate::physical_plan::DATA_FILES_COL_NAME;
use crate::physical_plan::delete::IsolationLevel;
use crate::to_datafusion_error;

/// Snapshot-summary key stamping every `IcebergCommitExec` commit with a unique id — the
/// ENGINE_CONTRACT §8 ambiguous-commit-outcome reconciliation class: on a transport-ambiguous
/// failure the engine reloads the table and scans recent snapshot summaries for this id BEFORE
/// re-running, so a retry can never silently duplicate an already-landed INSERT. The key matches the
/// one named in §8 (and the downstream RePark `OPERATION_ID_PROP`) so one reconciliation recipe
/// serves every engine surface.
pub(crate) const OPERATION_ID_PROP: &str = "engine.operation-id";

/// The isolation-level table property for `INSERT OVERWRITE` and its accepted values. ENGINE-DEFINED
/// (this crate), NOT an Iceberg-standard property: Java/Spark expose overwrite isolation only as a
/// per-write OPTION (`SparkWriteOptions.ISOLATION_LEVEL = "isolation-level"`, read via
/// `SparkWriteConf.isolationLevel()` `parseOptional` — absent by default, in which case Spark runs NO
/// overwrite validations, `SparkWrite.java` L364-377), and this DataFusion seam has no per-write
/// options. Values: `serializable` / `snapshot` (the §5 arms) / `none` (Spark's default absent-option
/// behavior — no validation). DEFAULT: `snapshot` — a deliberate, documented divergence from Spark's
/// unvalidated default, arming the §5 recipe against the concurrent-delete-loss class.
pub(crate) const WRITE_OVERWRITE_ISOLATION_LEVEL: &str = "write.overwrite.isolation-level";
const OVERWRITE_ISOLATION_NONE: &str = "none";

/// Resolve the `INSERT OVERWRITE` isolation policy from the table properties (see
/// [`WRITE_OVERWRITE_ISOLATION_LEVEL`]): `None` = validations off (`"none"`, Spark's absent-option
/// default); `Some(level)` = arm the §5 arms for that level. Default `Some(Snapshot)`. Resolved at
/// execute time, mirroring Java's `writeConf.isolationLevel()` read inside `commit()`.
fn overwrite_isolation_level(table: &Table) -> DFResult<Option<IsolationLevel>> {
    match table
        .metadata()
        .properties()
        .get(WRITE_OVERWRITE_ISOLATION_LEVEL)
    {
        None => Ok(Some(IsolationLevel::Snapshot)),
        Some(name) if name.eq_ignore_ascii_case(OVERWRITE_ISOLATION_NONE) => Ok(None),
        Some(name) => IsolationLevel::parse(name).map(Some),
    }
}

/// IcebergCommitExec is responsible for collecting the files written and committing them per the DML
/// write operation, stamping every produced snapshot with a unique [`OPERATION_ID_PROP`] (§8).
#[derive(Debug)]
pub(crate) struct IcebergCommitExec {
    table: Table,
    catalog: Arc<dyn Catalog>,
    input: Arc<dyn ExecutionPlan>,
    schema: ArrowSchemaRef,
    /// The DML write operation: `Append` commits via `fast_append` (no §5 validations — appends are
    /// conflict-free by construction, Java `SparkWrite.BatchAppend`); `Overwrite` (`INSERT OVERWRITE`)
    /// replaces ALL existing data via `overwrite_files().overwrite_by_row_filter(AlwaysTrue)` with the
    /// §5 static-overwrite validations per [`WRITE_OVERWRITE_ISOLATION_LEVEL`]. Both stamp
    /// [`OPERATION_ID_PROP`].
    insert_op: InsertOp,
    count_schema: ArrowSchemaRef,
    plan_properties: Arc<PlanProperties>,
    commit_branch: Option<String>,
}

impl IcebergCommitExec {
    pub fn new(
        table: Table,
        catalog: Arc<dyn Catalog>,
        input: Arc<dyn ExecutionPlan>,
        schema: ArrowSchemaRef,
        insert_op: InsertOp,
    ) -> Self {
        let count_schema = Self::make_count_schema();

        let plan_properties = Self::compute_properties(Arc::clone(&count_schema));

        Self {
            table,
            catalog,
            input,
            schema,
            insert_op,
            count_schema,
            plan_properties,
            commit_branch: None,
        }
    }

    pub(crate) fn with_commit_branch(mut self, branch: Option<String>) -> Self {
        self.commit_branch = branch;
        self
    }

    // Compute the plan properties for this execution plan
    fn compute_properties(schema: ArrowSchemaRef) -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ))
    }

    // Create a record batch with just the count of rows written
    fn make_count_batch(count: u64) -> DFResult<RecordBatch> {
        let count_array = Arc::new(UInt64Array::from(vec![count])) as ArrayRef;

        RecordBatch::try_from_iter_with_nullable(vec![("count", count_array, false)]).map_err(|e| {
            DataFusionError::ArrowError(
                Box::new(e),
                Some("Failed to make count batch!".to_string()),
            )
        })
    }

    fn make_count_schema() -> ArrowSchemaRef {
        // Define a schema.
        Arc::new(ArrowSchema::new(vec![Field::new(
            "count",
            DataType::UInt64,
            false,
        )]))
    }
}

impl DisplayAs for IcebergCommitExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(f, "IcebergCommitExec: table={}", self.table.identifier())
            }
            DisplayFormatType::Verbose => {
                write!(
                    f,
                    "IcebergCommitExec: table={}, schema={:?}",
                    self.table.identifier(),
                    self.schema
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "IcebergCommitExec: table={}", self.table.identifier())
            }
        }
    }
}

impl ExecutionPlan for IcebergCommitExec {
    fn name(&self) -> &str {
        "IcebergCommitExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn required_input_distribution(&self) -> Vec<datafusion::physical_plan::Distribution> {
        vec![datafusion::physical_plan::Distribution::SinglePartition; self.children().len()]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "IcebergCommitExec expects exactly one child, but provided {}",
                children.len()
            )));
        }

        Ok(Arc::new(
            IcebergCommitExec::new(
                self.table.clone(),
                self.catalog.clone(),
                children[0].clone(),
                self.schema.clone(),
                self.insert_op,
            )
            .with_commit_branch(self.commit_branch.clone()),
        ))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // IcebergCommitExec only has one partition (partition 0)
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "IcebergCommitExec only has one partition, but got partition {partition}"
            )));
        }

        let table = self.table.clone();
        let input_plan = self.input.clone();

        // todo revisit this
        let spec_id = self.table.metadata().default_partition_spec_id();
        let partition_type = self.table.metadata().default_partition_type().clone();
        let current_schema = self.table.metadata().current_schema().clone();

        let catalog = Arc::clone(&self.catalog);
        let insert_op = self.insert_op;
        let commit_branch = self.commit_branch.clone();

        // Process the input streams from all partitions and commit the data files
        let stream = futures::stream::once(async move {
            let mut data_files: Vec<DataFile> = Vec::new();
            let mut total_record_count: u64 = 0;

            // Execute and collect results from the input coalesced plan
            let mut batch_stream = input_plan.execute(0, context)?;

            while let Some(batch_result) = batch_stream.next().await {
                let batch = batch_result?;

                let files_array = batch
                    .column_by_name(DATA_FILES_COL_NAME)
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "Expected 'data_files' column in input batch".to_string(),
                        )
                    })?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "Expected 'data_files' column to be StringArray".to_string(),
                        )
                    })?;

                // Deserialize all data files from the StringArray
                let batch_files: Vec<DataFile> = files_array
                    .into_iter()
                    .flatten()
                    .map(|f| -> DFResult<DataFile> {
                        // Parse JSON to DataFileSerde and convert to DataFile
                        deserialize_data_file_from_json(
                            f,
                            spec_id,
                            &partition_type,
                            &current_schema,
                        )
                        .map_err(to_datafusion_error)
                    })
                    .collect::<datafusion::common::Result<_>>()?;

                // add record_counts from the current batch to total record count
                total_record_count += batch_files.iter().map(|f| f.record_count()).sum::<u64>();

                // Add all deserialized files to our collection
                data_files.extend(batch_files);
            }

            // NOTE (empty-commit semantics, BUG-001/BUG-004): there is deliberately NO
            // `if data_files.is_empty() { return empty }` short-circuit here. A blanket early
            // return silently no-ops an empty `INSERT OVERWRITE` — but Spark's static full-table
            // overwrite must WIPE every existing row even with zero result rows (Java
            // `SparkWrite.OverwriteByFilter.commit`, apache-iceberg 1.10.0 L354-384, commits
            // `overwriteByRowFilter(alwaysTrue)` UNCONDITIONALLY — unlike `DynamicOverwrite.commit`
            // L313-316 which alone skips on empty). It also skips the empty-`INSERT INTO` snapshot
            // that Java `SparkWrite.BatchAppend.commit` (L292-306) stamps unconditionally via
            // `table.newAppend()` (`SnapshotProducer.commit` always adds a fresh snapshot). Instead
            // every insert op runs its normal transaction below: empty Overwrite → delete-all wipe
            // in one atomic snapshot (with the §5 OCC validations); empty Append → empty-append
            // snapshot stamp. Both are enabled by the non-empty `OPERATION_ID_PROP` snapshot
            // property, which keeps the producer's "truly-empty commit" guard from rejecting a
            // no-added-files commit. The returned count batch is `total_record_count` (0 for an
            // empty write), consistent with the non-empty path — never a zero-row batch.

            // One unique operation id per statement execution (§8): stamped into the produced
            // snapshot's summary so an ambiguous commit outcome can be reconciled by scanning recent
            // summaries for this id before re-running. The id is action state, so the transaction's
            // internal refresh-re-apply loop reuses the SAME id — a retried attempt can never mint a
            // second stamp (the idempotency evidence stays unique).
            let operation_id = uuid::Uuid::new_v4().to_string();
            let snapshot_properties =
                HashMap::from([(OPERATION_ID_PROP.to_string(), operation_id)]);

            // Create a transaction and commit the data files per the DML write operation.
            let tx = Transaction::new(&table);
            let committed = match insert_op {
                // INSERT INTO — append the new data files. No §5 validations: an append neither
                // reads table state nor removes files, so nothing can conflict (Java
                // `SparkWrite.BatchAppend.commit` runs none).
                InsertOp::Append => {
                    let action = crate::physical_plan::snapshot_target::maybe_to_branch(
                        tx.fast_append()
                            .add_data_files(data_files)
                            .set_snapshot_properties(snapshot_properties),
                        commit_branch.as_deref(),
                        |action, branch| action.to_branch(branch),
                    );
                    action
                        .apply(tx)
                        .map_err(to_datafusion_error)?
                        .commit(catalog.as_ref())
                        .await
                }
                // INSERT OVERWRITE — replace ALL existing data: delete every live row (an
                // always-true overwrite filter removes all current data files) and add the new files
                // in one atomic snapshot. §5 static-overwrite recipe (Java
                // `SparkWrite.OverwriteByFilter.commit` L364-377): snapshot →
                // `validate_no_conflicting_deletes` (L374-375); serializable → +
                // `validate_no_conflicting_data` (L371-373). NO explicit conflict-detection filter —
                // Java never sets one here; the row filter itself is the default conflict filter.
                // `validate_from_snapshot` is armed with the handle's current snapshot (Java arms it
                // only when the writer tracked one, L367-369; this exec's natural anchor is the
                // table state the statement was planned against). The policy knob (incl. `none` =
                // Spark's unvalidated default) is documented on
                // [`WRITE_OVERWRITE_ISOLATION_LEVEL`].
                InsertOp::Overwrite => {
                    let mut action = tx
                        .overwrite_files()
                        .overwrite_by_row_filter(Predicate::AlwaysTrue)
                        .add_files(data_files)
                        .set_snapshot_properties(snapshot_properties);
                    if let Some(isolation) = overwrite_isolation_level(&table)? {
                        action = action.validate_no_conflicting_deletes();
                        if isolation == IsolationLevel::Serializable {
                            action = action.validate_no_conflicting_data();
                        }
                        action =
                            crate::physical_plan::snapshot_target::maybe_validate_from_snapshot(
                                action,
                                commit_branch.as_deref(),
                                table.metadata().current_snapshot_id(),
                                |action, snapshot_id| action.validate_from_snapshot(snapshot_id),
                            );
                    }
                    let action = crate::physical_plan::snapshot_target::maybe_to_branch(
                        action,
                        commit_branch.as_deref(),
                        |action, branch| action.to_branch(branch),
                    );
                    action
                        .apply(tx)
                        .map_err(to_datafusion_error)?
                        .commit(catalog.as_ref())
                        .await
                }
                // `Replace` (upsert/ON CONFLICT) has no single Iceberg commit primitive — out of scope.
                InsertOp::Replace => {
                    return Err(DataFusionError::NotImplemented(
                        "INSERT ... Replace (upsert) is not supported for Iceberg tables"
                            .to_string(),
                    ));
                }
            };
            committed.map_err(to_datafusion_error)?;

            Self::make_count_batch(total_record_count)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.count_schema),
            stream,
        )))
    }
}

#[cfg(test)]
#[path = "commit_tests.rs"]
mod tests;
