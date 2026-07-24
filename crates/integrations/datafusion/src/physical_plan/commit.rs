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

use std::any::Any;
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
    plan_properties: PlanProperties,
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
        }
    }

    // Compute the plan properties for this execution plan
    fn compute_properties(schema: ArrowSchemaRef) -> PlanProperties {
        PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        )
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
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

        Ok(Arc::new(IcebergCommitExec::new(
            self.table.clone(),
            self.catalog.clone(),
            children[0].clone(),
            self.schema.clone(),
            self.insert_op,
        )))
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
                    tx.fast_append()
                        .add_data_files(data_files)
                        .set_snapshot_properties(snapshot_properties)
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
                        if let Some(snapshot_id) = table.metadata().current_snapshot_id() {
                            action = action.validate_from_snapshot(snapshot_id);
                        }
                    }
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
mod tests {
    use std::collections::HashMap;
    use std::fmt;
    use std::sync::Arc;

    use datafusion::arrow::array::{ArrayRef, Int32Array, RecordBatch, StringArray, UInt64Array};
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::datasource::MemTable;
    use datafusion::execution::context::TaskContext;
    use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
    use datafusion::physical_plan::common::collect;
    use datafusion::physical_plan::execution_plan::Boundedness;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
    use datafusion::prelude::*;
    use futures::StreamExt;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, NestedField, PrimitiveType,
        Schema, Struct, Transform, Type, UnboundPartitionSpec,
    };
    use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

    use super::*;
    use crate::physical_plan::DATA_FILES_COL_NAME;
    use crate::table::IcebergTableProvider;

    // A mock execution plan that returns record batches with serialized data files
    #[derive(Debug)]
    struct MockWriteExec {
        schema: Arc<ArrowSchema>,
        data_files_json: Vec<String>,
        plan_properties: PlanProperties,
    }

    impl MockWriteExec {
        fn new(data_files_json: Vec<String>) -> Self {
            let schema = Arc::new(ArrowSchema::new(vec![Field::new(
                DATA_FILES_COL_NAME,
                DataType::Utf8,
                false,
            )]));

            let plan_properties = PlanProperties::new(
                EquivalenceProperties::new(schema.clone()),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            );

            Self {
                schema,
                data_files_json,
                plan_properties,
            }
        }
    }

    impl ExecutionPlan for MockWriteExec {
        fn name(&self) -> &str {
            "MockWriteExec"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn schema(&self) -> Arc<ArrowSchema> {
            self.schema.clone()
        }

        fn properties(&self) -> &PlanProperties {
            &self.plan_properties
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }

        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> datafusion::common::Result<SendableRecordBatchStream> {
            // Create a record batch with the serialized data files
            let array = Arc::new(StringArray::from(self.data_files_json.clone())) as ArrayRef;
            let batch = RecordBatch::try_new(self.schema.clone(), vec![array])?;

            // Create a stream that returns this batch
            let stream = futures::stream::once(async move { Ok(batch) }).boxed();
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema(),
                stream,
            )))
        }
    }

    // Implement DisplayAs for MockDataFilesExec
    impl DisplayAs for MockWriteExec {
        fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
            match t {
                DisplayFormatType::Default
                | DisplayFormatType::Verbose
                | DisplayFormatType::TreeRender => {
                    write!(f, "MockDataFilesExec: files={}", self.data_files_json.len())
                }
            }
        }
    }

    #[tokio::test]
    async fn test_iceberg_commit_exec() -> Result<(), Box<dyn std::error::Error>> {
        // Create a memory catalog with in-memory file IO
        let catalog = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        "memory://root".to_string(),
                    )]),
                )
                .await
                .unwrap(),
        );

        // Create a namespace
        let namespace = NamespaceIdent::new("test_namespace".to_string());
        catalog.create_namespace(&namespace, HashMap::new()).await?;

        // Create a schema for the table
        let schema = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()?;

        // Create a table
        let table_creation = TableCreation::builder()
            .name("test_table".to_string())
            .schema(schema)
            .location("memory://root/test_table".to_string())
            .properties(HashMap::new())
            .build();

        let table = catalog.create_table(&namespace, table_creation).await?;

        // Create data files
        let data_file1 = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("path/to/file1.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1024)
            .record_count(100)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::empty())
            .build()?;

        let data_file2 = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("path/to/file2.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(2048)
            .record_count(200)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::empty())
            .build()?;

        // Serialize data files to JSON
        let partition_type = table.metadata().default_partition_type().clone();
        let data_file1_json = iceberg::spec::serialize_data_file_to_json(
            data_file1.clone(),
            &partition_type,
            table.metadata().format_version(),
        )?;

        let data_file2_json = iceberg::spec::serialize_data_file_to_json(
            data_file2.clone(),
            &partition_type,
            table.metadata().format_version(),
        )?;

        // Create a mock execution plan that returns the serialized data files
        let input_exec = Arc::new(MockWriteExec::new(vec![data_file1_json, data_file2_json]));

        // Create the IcebergCommitExec
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            DATA_FILES_COL_NAME,
            DataType::Utf8,
            false,
        )]));

        let commit_exec = IcebergCommitExec::new(
            table.clone(),
            catalog.clone(),
            input_exec,
            arrow_schema,
            InsertOp::Append,
        );

        // Verify Execution Plan schema matches the count schema
        assert_eq!(commit_exec.schema(), IcebergCommitExec::make_count_schema());

        // Execute the commit exec
        let task_ctx = Arc::new(TaskContext::default());
        let stream = commit_exec.execute(0, task_ctx)?;
        let batches = collect(stream).await?;

        // Verify the results
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), 1);

        // The output should be a record batch with a single column "count" and a single row
        // with the total record count (100 + 200 = 300)
        let count_array = batch.column(0);
        assert_eq!(count_array.len(), 1);
        assert_eq!(count_array.data_type(), &DataType::UInt64);

        // Verify that the count is correct
        let count = count_array.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(count.value(0), 300);

        // Verify that the table has been updated with the new files
        let updated_table = catalog
            .load_table(&TableIdent::from_strs(["test_namespace", "test_table"]).unwrap())
            .await?;
        let current_snapshot = updated_table.metadata().current_snapshot().unwrap();

        // Load the manifest list to verify the data files were added
        let manifest_list = current_snapshot
            .load_manifest_list(updated_table.file_io(), updated_table.metadata())
            .await?;

        // There should be at least one manifest
        assert!(!manifest_list.entries().is_empty());

        // Load the first manifest and verify it contains our data files
        let manifest = manifest_list.entries()[0]
            .load_manifest(updated_table.file_io())
            .await?;

        // Verify that the manifest contains our data files
        let manifest_files: Vec<String> = manifest
            .entries()
            .iter()
            .map(|entry| entry.data_file().file_path().to_string())
            .collect();

        assert!(manifest_files.contains(&"path/to/file1.parquet".to_string()));
        assert!(manifest_files.contains(&"path/to/file2.parquet".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_datafusion_execution_partitioned_source() -> Result<(), Box<dyn std::error::Error>>
    {
        let catalog = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        "memory://root".to_string(),
                    )]),
                )
                .await?,
        );

        let namespace = NamespaceIdent::new("test_namespace".to_string());
        catalog.create_namespace(&namespace, HashMap::new()).await?;

        let schema = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()?;

        let table_name = "test_table";
        let table_creation = TableCreation::builder()
            .name(table_name.to_string())
            .schema(schema)
            .location("memory://root/test_table".to_string())
            .properties(HashMap::new())
            .build();
        let _ = catalog.create_table(&namespace, table_creation).await?;

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batches: Vec<RecordBatch> = (1..4)
            .map(|idx| {
                RecordBatch::try_new(arrow_schema.clone(), vec![
                    Arc::new(Int32Array::from(vec![idx])) as ArrayRef,
                    Arc::new(StringArray::from(vec![format!("Name{idx}")])) as ArrayRef,
                ])
            })
            .collect::<Result<_, _>>()?;

        // Create DataFusion context with specific partition configuration
        let mut config = SessionConfig::new();
        config = config.set_usize("datafusion.execution.target_partitions", 8);
        let ctx = SessionContext::new_with_config(config);

        // Create multiple partitions - each batch becomes a separate partition
        let partitions: Vec<Vec<RecordBatch>> =
            batches.into_iter().map(|batch| vec![batch]).collect();
        let source_table = Arc::new(MemTable::try_new(Arc::clone(&arrow_schema), partitions)?);
        ctx.register_table("source_table", source_table)?;

        let iceberg_table_provider = IcebergTableProvider::try_new(
            catalog.clone(),
            namespace.clone(),
            table_name.to_string(),
        )
        .await?;
        ctx.register_table("iceberg_table", Arc::new(iceberg_table_provider))?;

        let insert_plan = ctx
            .sql("INSERT INTO iceberg_table SELECT * FROM source_table")
            .await?;

        let physical_plan = insert_plan.create_physical_plan().await?;

        let actual_plan = format!(
            "{}",
            datafusion::physical_plan::displayable(physical_plan.as_ref()).indent(false)
        );

        println!("Physical plan:\n{actual_plan}");

        let expected_plan = "\
IcebergCommitExec: table=test_namespace.test_table
  CoalescePartitionsExec
    IcebergWriteExec: table=test_namespace.test_table
      DataSourceExec: partitions=3, partition_sizes=[1, 1, 1]";

        assert_eq!(
            actual_plan.trim(),
            expected_plan.trim(),
            "Physical plan does not match expected\n\nExpected:\n{}\n\nActual:\n{}",
            expected_plan.trim(),
            actual_plan.trim()
        );

        Ok(())
    }

    // ============================================================================================
    // Empty-commit semantics (BUG-001 Critical / BUG-004): an empty `INSERT OVERWRITE` must WIPE
    // the table (Java `SparkWrite.OverwriteByFilter.commit` commits `overwriteByRowFilter(alwaysTrue)`
    // unconditionally, apache-iceberg 1.10.0), and an empty `INSERT INTO` must still stamp an
    // (empty) Append snapshot (Java `SparkWrite.BatchAppend.commit`). Non-empty paths stay unchanged.
    // ============================================================================================

    /// A boxed-error result for the test helpers/bodies below.
    type BoxResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    type TestResult = BoxResult<()>;

    /// Create a memory catalog + an unpartitioned `(id int, name string)` table with `props`.
    async fn setup_table(props: HashMap<String, String>) -> BoxResult<(Arc<dyn Catalog>, Table)> {
        let catalog: Arc<dyn Catalog> = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        "memory://root".to_string(),
                    )]),
                )
                .await?,
        );
        let namespace = NamespaceIdent::new("ns".to_string());
        catalog.create_namespace(&namespace, HashMap::new()).await?;
        let schema = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()?;
        let table_creation = TableCreation::builder()
            .name("t".to_string())
            .schema(schema)
            .location("memory://root/t".to_string())
            .properties(props)
            .build();
        let table = catalog.create_table(&namespace, table_creation).await?;
        Ok((catalog, table))
    }

    /// Build an unpartitioned metadata-only [`DataFile`] at `path` carrying `record_count` rows.
    fn make_data_file(table: &Table, path: &str, record_count: u64) -> BoxResult<DataFile> {
        Ok(DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1024)
            .record_count(record_count)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::empty())
            .build()?)
    }

    /// Serialize a [`DataFile`] to the JSON the write→commit seam exchanges (what `MockWriteExec` emits).
    fn data_file_json(table: &Table, file: DataFile) -> BoxResult<String> {
        let partition_type = table.metadata().default_partition_type().clone();
        Ok(iceberg::spec::serialize_data_file_to_json(
            file,
            &partition_type,
            table.metadata().format_version(),
        )?)
    }

    /// Append `files` via a direct `fast_append` transaction (bypassing the commit exec) and return
    /// the refreshed table — used both to pre-populate a table and to simulate a concurrent commit.
    async fn append_files_direct(
        catalog: &Arc<dyn Catalog>,
        table: &Table,
        files: Vec<DataFile>,
    ) -> BoxResult<Table> {
        let tx = Transaction::new(table);
        let action = tx.fast_append().add_data_files(files);
        let tx = action.apply(tx)?;
        Ok(tx.commit(catalog.as_ref()).await?)
    }

    /// Run an [`IcebergCommitExec`] over a `MockWriteExec` emitting `files_json` and collect its
    /// output batches (the row-count batch), or the commit error.
    async fn run_commit_exec(
        table: &Table,
        catalog: &Arc<dyn Catalog>,
        files_json: Vec<String>,
        insert_op: InsertOp,
    ) -> DFResult<Vec<RecordBatch>> {
        let input = Arc::new(MockWriteExec::new(files_json));
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            DATA_FILES_COL_NAME,
            DataType::Utf8,
            false,
        )]));
        let exec = IcebergCommitExec::new(
            table.clone(),
            Arc::clone(catalog),
            input,
            arrow_schema,
            insert_op,
        );
        let stream = exec.execute(0, Arc::new(TaskContext::default()))?;
        collect(stream).await
    }

    /// The sorted set of LIVE data-file paths reachable from `snapshot` — the real correctness
    /// signal (what a scan would read).
    async fn live_paths_in(
        table: &Table,
        snapshot: &iceberg::spec::SnapshotRef,
    ) -> BoxResult<Vec<String>> {
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await?;
        let mut paths = Vec::new();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await?;
            for entry in manifest.entries() {
                if entry.is_alive() {
                    paths.push(entry.file_path().to_string());
                }
            }
        }
        paths.sort();
        Ok(paths)
    }

    /// Assert the single row-count batch reports `expected` rows written (the "rows added" count
    /// semantics — an empty write reports 0, never a zero-row batch).
    fn assert_count(batches: &[RecordBatch], expected: u64) {
        assert_eq!(batches.len(), 1, "commit emits exactly one count batch");
        let batch = &batches[0];
        assert_eq!(
            batch.num_rows(),
            1,
            "count batch has exactly one row (not empty)"
        );
        let count = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("count column is UInt64");
        assert_eq!(count.value(0), expected, "row-count value");
    }

    /// THE load-bearing Critical pin (BUG-001). A table holds three live rows; an EMPTY
    /// `INSERT OVERWRITE` (zero result rows) must WIPE the table in one new snapshot — the prior
    /// data must be gone from the current snapshot, a NEW snapshot must exist, and the PRIOR
    /// snapshot must remain intact (time-travel). Before the fix the empty write short-circuited
    /// and the old rows silently survived.
    #[tokio::test]
    async fn test_empty_overwrite_wipes_table_bug001() -> TestResult {
        let (catalog, table) = setup_table(HashMap::new()).await?;

        // Pre-populate: A, B, C live (snapshot S0).
        let table = append_files_direct(&catalog, &table, vec![
            make_data_file(&table, "a.parquet", 100)?,
            make_data_file(&table, "b.parquet", 200)?,
            make_data_file(&table, "c.parquet", 300)?,
        ])
        .await?;
        let prior_snapshot_id = table
            .metadata()
            .current_snapshot_id()
            .expect("S0 exists after append");
        let prior_snapshot_count = table.metadata().snapshots().len();
        assert_eq!(
            live_paths_in(&table, table.metadata().current_snapshot().expect("S0")).await?,
            vec!["a.parquet", "b.parquet", "c.parquet"],
            "precondition: three rows live before the overwrite"
        );

        // Empty INSERT OVERWRITE against the table handle at S0.
        let batches = run_commit_exec(&table, &catalog, vec![], InsertOp::Overwrite).await?;
        assert_count(&batches, 0);

        // Reload and assert the table is WIPED: zero live files under a NEW snapshot.
        let reloaded = catalog
            .load_table(&TableIdent::from_strs(["ns", "t"])?)
            .await?;
        let new_snapshot = reloaded
            .metadata()
            .current_snapshot()
            .expect("a new snapshot exists after the overwrite");
        assert_ne!(
            new_snapshot.snapshot_id(),
            prior_snapshot_id,
            "the empty overwrite produced a NEW snapshot"
        );
        assert_eq!(
            reloaded.metadata().snapshots().len(),
            prior_snapshot_count + 1,
            "exactly one snapshot was added"
        );
        assert!(
            live_paths_in(&reloaded, new_snapshot).await?.is_empty(),
            "the empty overwrite WIPED every live row (scan returns 0 rows)"
        );

        // Time-travel: the PRIOR snapshot is intact and still sees the original three rows.
        let prior = reloaded
            .metadata()
            .snapshot_by_id(prior_snapshot_id)
            .expect("prior snapshot still resolvable");
        assert_eq!(
            live_paths_in(&reloaded, prior).await?,
            vec!["a.parquet", "b.parquet", "c.parquet"],
            "the prior snapshot's data is preserved for time-travel"
        );

        Ok(())
    }

    /// §5 OCC preservation for the empty-overwrite path. With `write.overwrite.isolation-level =
    /// serializable`, an empty overwrite whose transaction started at S0 must still FAIL LOUD when a
    /// concurrent commit added data after S0 (Java `validateNoConflictingData` armed by
    /// `OverwriteByFilter.commit`) — and, having failed, must NOT have destroyed the concurrently
    /// added data. This proves the wipe fix did not disarm the isolation validations.
    #[tokio::test]
    async fn test_empty_overwrite_preserves_serializable_occ_validation() -> TestResult {
        let (catalog, table) = setup_table(HashMap::from([(
            WRITE_OVERWRITE_ISOLATION_LEVEL.to_string(),
            "serializable".to_string(),
        )]))
        .await?;

        // Base row A (snapshot S0); the exec captures this handle.
        let table_at_s0 = append_files_direct(&catalog, &table, vec![make_data_file(
            &table,
            "a.parquet",
            100,
        )?])
        .await?;

        // Concurrent commit AFTER S0: append B (advances the catalog head to S1).
        append_files_direct(&catalog, &table_at_s0, vec![make_data_file(
            &table_at_s0,
            "b.parquet",
            200,
        )?])
        .await?;

        // Empty overwrite whose transaction still starts at S0 → serializable conflict with B.
        let result = run_commit_exec(&table_at_s0, &catalog, vec![], InsertOp::Overwrite).await;
        assert!(
            result.is_err(),
            "empty overwrite under SERIALIZABLE must reject a concurrent add since its start snapshot"
        );

        // The failed overwrite must NOT have wiped the table: both rows are still live.
        let reloaded = catalog
            .load_table(&TableIdent::from_strs(["ns", "t"])?)
            .await?;
        assert_eq!(
            live_paths_in(
                &reloaded,
                reloaded.metadata().current_snapshot().expect("head")
            )
            .await?,
            vec!["a.parquet", "b.parquet"],
            "a rejected serializable overwrite preserves the concurrently added data (no data loss)"
        );

        Ok(())
    }

    /// BUG-004 pin (per the Java oracle: Java `SparkWrite.BatchAppend.commit` → `table.newAppend()`
    /// commits unconditionally, and `SnapshotProducer.commit` always stamps a fresh snapshot). An
    /// empty `INSERT INTO` must therefore stamp an (empty) Append snapshot rather than no-op. The
    /// non-empty `OPERATION_ID_PROP` snapshot property keeps the producer's truly-empty-commit guard
    /// from rejecting the no-added-files append.
    #[tokio::test]
    async fn test_empty_append_stamps_snapshot_bug004() -> TestResult {
        let (catalog, table) = setup_table(HashMap::new()).await?;
        assert!(
            table.metadata().current_snapshot().is_none(),
            "precondition: brand-new table has no snapshot"
        );

        let batches = run_commit_exec(&table, &catalog, vec![], InsertOp::Append).await?;
        assert_count(&batches, 0);

        let reloaded = catalog
            .load_table(&TableIdent::from_strs(["ns", "t"])?)
            .await?;
        let snapshot = reloaded
            .metadata()
            .current_snapshot()
            .expect("empty append stamped a snapshot (Java parity)");
        assert_eq!(
            snapshot.summary().operation,
            iceberg::spec::Operation::Append,
            "the empty stamp is an Append snapshot"
        );
        assert!(
            snapshot
                .summary()
                .additional_properties
                .contains_key(OPERATION_ID_PROP),
            "the empty append still carries the §8 operation-id stamp"
        );
        assert!(
            live_paths_in(&reloaded, snapshot).await?.is_empty(),
            "the empty append adds no data files"
        );
        Ok(())
    }

    /// Regression guard: a NON-empty `INSERT OVERWRITE` is unchanged by the fix — it replaces all
    /// existing data with the new files in one Overwrite snapshot and reports the added row count.
    #[tokio::test]
    async fn test_nonempty_overwrite_replaces_all_data() -> TestResult {
        let (catalog, table) = setup_table(HashMap::new()).await?;
        let table = append_files_direct(&catalog, &table, vec![
            make_data_file(&table, "old_a.parquet", 100)?,
            make_data_file(&table, "old_b.parquet", 200)?,
        ])
        .await?;

        let new_json = data_file_json(&table, make_data_file(&table, "new.parquet", 42)?)?;
        let batches =
            run_commit_exec(&table, &catalog, vec![new_json], InsertOp::Overwrite).await?;
        assert_count(&batches, 42);

        let reloaded = catalog
            .load_table(&TableIdent::from_strs(["ns", "t"])?)
            .await?;
        let snapshot = reloaded.metadata().current_snapshot().expect("head");
        assert_eq!(
            snapshot.summary().operation,
            iceberg::spec::Operation::Overwrite,
            "a non-empty full overwrite records Overwrite"
        );
        assert_eq!(
            live_paths_in(&reloaded, snapshot).await?,
            vec!["new.parquet"],
            "non-empty overwrite replaces all prior data with exactly the new file"
        );
        Ok(())
    }

    // ============================================================================================
    // NOVEL PIN (partitioned table): the empty-overwrite wipe must cut across EVERY partition in a
    // single snapshot — the always-true row filter of `OverwriteByFilter.commit` removes all live
    // data files regardless of partition value (Java `SparkWrite.OverwriteByFilter`, unconditional
    // `overwriteByRowFilter(alwaysTrue)`). The same partitioned table must NOT be wiped by an empty
    // `INSERT INTO` (that only stamps), and a non-empty overwrite must still replace all partitions.
    // ============================================================================================

    /// Create a memory catalog + an `(id int, name string)` table partitioned by `identity(id)`.
    async fn setup_partitioned_table(
        props: HashMap<String, String>,
    ) -> BoxResult<(Arc<dyn Catalog>, Table)> {
        let catalog: Arc<dyn Catalog> = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        "memory://root".to_string(),
                    )]),
                )
                .await?,
        );
        let namespace = NamespaceIdent::new("ns".to_string());
        catalog.create_namespace(&namespace, HashMap::new()).await?;
        let schema = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()?;
        // Identity partition on `id` (source field id = 1): each distinct id is its own partition.
        let partition_spec = UnboundPartitionSpec::builder()
            .add_partition_field(1, "id", Transform::Identity)?
            .build();
        let table_creation = TableCreation::builder()
            .name("tp".to_string())
            .schema(schema)
            .location("memory://root/tp".to_string())
            .partition_spec(partition_spec)
            .properties(props)
            .build();
        let table = catalog.create_table(&namespace, table_creation).await?;
        Ok((catalog, table))
    }

    /// Build a metadata-only [`DataFile`] routed to the identity partition `id = part_id`.
    fn make_partitioned_data_file(
        table: &Table,
        path: &str,
        part_id: i32,
        record_count: u64,
    ) -> BoxResult<DataFile> {
        Ok(DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1024)
            .record_count(record_count)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::int(part_id))]))
            .build()?)
    }

    /// NOVEL: on a table partitioned by `identity(id)` with TWO populated partitions:
    ///   (1) an empty `INSERT INTO` must NOT wipe — both partitions stay live and a snapshot is
    ///       stamped; then
    ///   (2) an empty `INSERT OVERWRITE` must WIPE EVERY partition in ONE new snapshot; and, on a
    ///       fresh two-partition table,
    ///   (3) a non-empty overwrite must replace ALL partitions with exactly the new file.
    #[tokio::test]
    async fn test_empty_overwrite_wipes_all_partitions_partitioned() -> TestResult {
        let (catalog, table) = setup_partitioned_table(HashMap::new()).await?;

        // Two distinct partitions live: id=1 (p1.parquet) and id=2 (p2.parquet).
        let table = append_files_direct(&catalog, &table, vec![
            make_partitioned_data_file(&table, "p1.parquet", 1, 100)?,
            make_partitioned_data_file(&table, "p2.parquet", 2, 200)?,
        ])
        .await?;
        let after_seed_snapshots = table.metadata().snapshots().len();
        assert_eq!(
            live_paths_in(&table, table.metadata().current_snapshot().expect("seed")).await?,
            vec!["p1.parquet", "p2.parquet"],
            "precondition: two partitions live"
        );

        // (1) Empty INSERT INTO append must NOT wipe — both partitions stay live, snapshot stamped.
        let batches = run_commit_exec(&table, &catalog, vec![], InsertOp::Append).await?;
        assert_count(&batches, 0);
        let table = catalog
            .load_table(&TableIdent::from_strs(["ns", "tp"])?)
            .await?;
        assert_eq!(
            table.metadata().snapshots().len(),
            after_seed_snapshots + 1,
            "empty append stamps exactly one new snapshot"
        );
        assert_eq!(
            live_paths_in(
                &table,
                table.metadata().current_snapshot().expect("post-append")
            )
            .await?,
            vec!["p1.parquet", "p2.parquet"],
            "empty append does NOT wipe any partition"
        );

        // (2) Empty INSERT OVERWRITE must WIPE EVERY partition in one new snapshot.
        let prior_snapshot_id = table.metadata().current_snapshot_id().expect("head");
        let prior_snapshots = table.metadata().snapshots().len();
        let batches = run_commit_exec(&table, &catalog, vec![], InsertOp::Overwrite).await?;
        assert_count(&batches, 0);
        let wiped = catalog
            .load_table(&TableIdent::from_strs(["ns", "tp"])?)
            .await?;
        let new_snapshot = wiped
            .metadata()
            .current_snapshot()
            .expect("overwrite stamps a snapshot");
        assert_ne!(
            new_snapshot.snapshot_id(),
            prior_snapshot_id,
            "the empty overwrite produced a NEW snapshot"
        );
        assert_eq!(
            wiped.metadata().snapshots().len(),
            prior_snapshots + 1,
            "exactly one snapshot added by the wipe"
        );
        assert!(
            live_paths_in(&wiped, new_snapshot).await?.is_empty(),
            "the empty overwrite WIPED EVERY partition (both p1 and p2 gone) in one snapshot"
        );

        // (3) Fresh two-partition table: a NON-empty overwrite replaces ALL partitions.
        let (catalog2, table2) = setup_partitioned_table(HashMap::new()).await?;
        let table2 = append_files_direct(&catalog2, &table2, vec![
            make_partitioned_data_file(&table2, "old_p1.parquet", 1, 100)?,
            make_partitioned_data_file(&table2, "old_p2.parquet", 2, 200)?,
        ])
        .await?;
        let new_json = data_file_json(
            &table2,
            make_partitioned_data_file(&table2, "new_p3.parquet", 3, 42)?,
        )?;
        let batches =
            run_commit_exec(&table2, &catalog2, vec![new_json], InsertOp::Overwrite).await?;
        assert_count(&batches, 42);
        let reloaded2 = catalog2
            .load_table(&TableIdent::from_strs(["ns", "tp"])?)
            .await?;
        assert_eq!(
            live_paths_in(
                &reloaded2,
                reloaded2.metadata().current_snapshot().expect("head")
            )
            .await?,
            vec!["new_p3.parquet"],
            "non-empty overwrite drops old partitions and keeps exactly the new file"
        );

        Ok(())
    }
}
