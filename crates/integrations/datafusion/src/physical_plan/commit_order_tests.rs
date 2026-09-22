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
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch, StringArray, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::config::ConfigOptions;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::execution::config::SessionConfig;
use datafusion::execution::context::TaskContext;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::common::collect;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::StreamExt;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, NestedField,
    PrimitiveType, Schema, Struct, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

use super::DataFileCommitOrder;
use crate::physical_plan::commit::IcebergCommitExec;
use crate::physical_plan::{DATA_FILES_COL_NAME, WRITE_PARTITION_INDEX_COL_NAME};

type BoxResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type TestResult = BoxResult<()>;

#[derive(Debug)]
struct MockWriteExec {
    schema: Arc<ArrowSchema>,
    files: Vec<(String, u64)>,
    plan_properties: Arc<PlanProperties>,
}

impl MockWriteExec {
    fn new(files: Vec<(String, u64)>) -> Self {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(DATA_FILES_COL_NAME, DataType::Utf8, false),
            Field::new(WRITE_PARTITION_INDEX_COL_NAME, DataType::UInt64, false),
        ]));
        let plan_properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            schema,
            files,
            plan_properties,
        }
    }
}

impl ExecutionPlan for MockWriteExec {
    fn name(&self) -> &str {
        "MockWriteExec"
    }

    fn schema(&self) -> Arc<ArrowSchema> {
        self.schema.clone()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
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
        let json: Vec<String> = self.files.iter().map(|(j, _)| j.clone()).collect();
        let index: Vec<u64> = self.files.iter().map(|(_, i)| *i).collect();
        let array = Arc::new(StringArray::from(json)) as ArrayRef;
        let index = Arc::new(UInt64Array::from(index)) as ArrayRef;
        let batch = RecordBatch::try_new(self.schema.clone(), vec![array, index])?;
        let stream = futures::stream::once(async move { Ok(batch) }).boxed();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}

impl DisplayAs for MockWriteExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "MockWriteExec: files={}", self.files.len())
            }
        }
    }
}

async fn memory_catalog() -> BoxResult<Arc<dyn Catalog>> {
    Ok(Arc::new(
        MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    "memory://root".to_string(),
                )]),
            )
            .await?,
    ))
}

async fn setup_partitioned_table() -> BoxResult<(Arc<dyn Catalog>, Table)> {
    let catalog = memory_catalog().await?;
    let namespace = NamespaceIdent::new("ns".to_string());
    catalog.create_namespace(&namespace, HashMap::new()).await?;
    let schema = Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    let partition_spec = UnboundPartitionSpec::builder()
        .add_partition_field(1, "id", Transform::Identity)?
        .build();
    let table_creation = TableCreation::builder()
        .name("tp".to_string())
        .schema(schema)
        .location("memory://root/tp".to_string())
        .partition_spec(partition_spec)
        .properties(HashMap::new())
        .build();
    let table = catalog.create_table(&namespace, table_creation).await?;
    Ok((catalog, table))
}

async fn setup_unpartitioned_table() -> BoxResult<(Arc<dyn Catalog>, Table)> {
    let catalog = memory_catalog().await?;
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
        .properties(HashMap::new())
        .build();
    let table = catalog.create_table(&namespace, table_creation).await?;
    Ok((catalog, table))
}

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

fn data_file_json(table: &Table, file: DataFile) -> BoxResult<String> {
    let partition_type = table.metadata().default_partition_type().clone();
    Ok(iceberg::spec::serialize_data_file_to_json(
        file,
        &partition_type,
        table.metadata().format_version(),
    )?)
}

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

async fn run_commit(
    table: &Table,
    catalog: &Arc<dyn Catalog>,
    files: Vec<(String, u64)>,
    session_config: SessionConfig,
) -> DFResult<Vec<RecordBatch>> {
    let input = Arc::new(MockWriteExec::new(files));
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
        InsertOp::Append,
        table.metadata().default_partition_spec().clone(),
    );
    let task_ctx = Arc::new(TaskContext::default().with_session_config(session_config));
    let stream = exec.execute(0, task_ctx)?;
    collect(stream).await
}

async fn ordered_live_paths(table: &Table) -> BoxResult<Vec<String>> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot");
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
    Ok(paths)
}

fn assert_count(batches: &[RecordBatch], expected: u64) {
    assert_eq!(batches.len(), 1, "commit emits exactly one count batch");
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1, "count batch has exactly one row");
    let count = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("count column is UInt64");
    assert_eq!(count.value(0), expected, "row-count value");
}

fn assert_permutation_refused(err: DataFusionError) {
    match err {
        DataFusionError::Internal(message) => assert_eq!(
            message,
            "data file commit order hook must return a permutation of its input files"
        ),
        other => panic!("expected Internal permutation error, got {other:?}"),
    }
}

fn scrambled_three(table: &Table) -> BoxResult<Vec<(String, u64)>> {
    Ok(vec![
        (
            data_file_json(
                table,
                make_partitioned_data_file(table, "p3.parquet", 3, 30)?,
            )?,
            2,
        ),
        (
            data_file_json(
                table,
                make_partitioned_data_file(table, "p1.parquet", 1, 10)?,
            )?,
            0,
        ),
        (
            data_file_json(
                table,
                make_partitioned_data_file(table, "p2.parquet", 2, 20)?,
            )?,
            1,
        ),
    ])
}

#[tokio::test]
async fn no_hook_commits_three_partitions_in_ascending_order() -> TestResult {
    let (catalog, table) = setup_partitioned_table().await?;
    let batches = run_commit(
        &table,
        &catalog,
        scrambled_three(&table)?,
        SessionConfig::new(),
    )
    .await?;
    assert_count(&batches, 60);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "tp"])?)
        .await?;
    assert_eq!(ordered_live_paths(&reloaded).await?, vec![
        "p1.parquet",
        "p2.parquet",
        "p3.parquet"
    ]);
    Ok(())
}

#[tokio::test]
async fn reverse_hook_commits_three_partitions_in_descending_order() -> TestResult {
    let (catalog, table) = setup_partitioned_table().await?;
    let order = DataFileCommitOrder::new(|mut files: Vec<DataFile>, _options: &ConfigOptions| {
        files.reverse();
        files
    });
    let config = SessionConfig::new().with_extension(Arc::new(order));
    let batches = run_commit(&table, &catalog, scrambled_three(&table)?, config).await?;
    assert_count(&batches, 60);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "tp"])?)
        .await?;
    assert_eq!(ordered_live_paths(&reloaded).await?, vec![
        "p3.parquet",
        "p2.parquet",
        "p1.parquet"
    ]);
    Ok(())
}

#[tokio::test]
async fn hook_reads_target_partitions_from_config_options() -> TestResult {
    let (catalog, table) = setup_partitioned_table().await?;
    let order = DataFileCommitOrder::new(|mut files: Vec<DataFile>, options: &ConfigOptions| {
        let shift = options.execution.target_partitions % files.len();
        files.rotate_left(shift);
        files
    });
    let config = SessionConfig::new()
        .set_usize("datafusion.execution.target_partitions", 8)
        .with_extension(Arc::new(order));
    let batches = run_commit(&table, &catalog, scrambled_three(&table)?, config).await?;
    assert_count(&batches, 60);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "tp"])?)
        .await?;
    assert_eq!(ordered_live_paths(&reloaded).await?, vec![
        "p3.parquet",
        "p1.parquet",
        "p2.parquet"
    ]);
    Ok(())
}

#[tokio::test]
async fn no_hook_unpartitioned_single_file_unchanged() -> TestResult {
    let (catalog, table) = setup_unpartitioned_table().await?;
    let files = vec![(
        data_file_json(&table, make_data_file(&table, "only.parquet", 7)?)?,
        0,
    )];
    let batches = run_commit(&table, &catalog, files, SessionConfig::new()).await?;
    assert_count(&batches, 7);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "t"])?)
        .await?;
    assert_eq!(ordered_live_paths(&reloaded).await?, vec!["only.parquet"]);
    Ok(())
}

#[tokio::test]
async fn drop_hook_fails_and_commits_no_snapshot() -> TestResult {
    let (catalog, table) = setup_partitioned_table().await?;
    let table = append_files_direct(&catalog, &table, vec![make_partitioned_data_file(
        &table,
        "seed.parquet",
        0,
        5,
    )?])
    .await?;
    let seed_snapshot = table
        .metadata()
        .current_snapshot_id()
        .expect("seed snapshot exists");
    let order = DataFileCommitOrder::new(|files: Vec<DataFile>, _options: &ConfigOptions| {
        files.into_iter().take(1).collect::<Vec<_>>()
    });
    let config = SessionConfig::new().with_extension(Arc::new(order));
    let new_files = vec![
        (
            data_file_json(
                &table,
                make_partitioned_data_file(&table, "n1.parquet", 1, 10)?,
            )?,
            0,
        ),
        (
            data_file_json(
                &table,
                make_partitioned_data_file(&table, "n2.parquet", 2, 20)?,
            )?,
            1,
        ),
    ];
    let err = run_commit(&table, &catalog, new_files, config)
        .await
        .expect_err("a hook that drops a file must fail the commit");
    assert_permutation_refused(err);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "tp"])?)
        .await?;
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        Some(seed_snapshot)
    );
    assert_eq!(ordered_live_paths(&reloaded).await?, vec!["seed.parquet"]);
    Ok(())
}

#[tokio::test]
async fn duplicate_hook_fails_and_commits_no_snapshot() -> TestResult {
    let (catalog, table) = setup_partitioned_table().await?;
    let table = append_files_direct(&catalog, &table, vec![make_partitioned_data_file(
        &table,
        "seed.parquet",
        0,
        5,
    )?])
    .await?;
    let seed_snapshot = table
        .metadata()
        .current_snapshot_id()
        .expect("seed snapshot exists");
    let order = DataFileCommitOrder::new(|files: Vec<DataFile>, _options: &ConfigOptions| {
        let mut out = files.clone();
        if out.len() == 2 {
            out[1] = out[0].clone();
        }
        out
    });
    let config = SessionConfig::new().with_extension(Arc::new(order));
    let new_files = vec![
        (
            data_file_json(
                &table,
                make_partitioned_data_file(&table, "n1.parquet", 1, 10)?,
            )?,
            0,
        ),
        (
            data_file_json(
                &table,
                make_partitioned_data_file(&table, "n2.parquet", 2, 20)?,
            )?,
            1,
        ),
    ];
    let err = run_commit(&table, &catalog, new_files, config)
        .await
        .expect_err("a hook that duplicates a file must fail the commit");
    assert_permutation_refused(err);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "tp"])?)
        .await?;
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        Some(seed_snapshot)
    );
    assert_eq!(ordered_live_paths(&reloaded).await?, vec!["seed.parquet"]);
    Ok(())
}
