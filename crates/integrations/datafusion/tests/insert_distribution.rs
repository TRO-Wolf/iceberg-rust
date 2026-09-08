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

use std::collections::{BTreeMap, HashMap};
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use anyhow::Result;
use datafusion::arrow::array::{
    Array, Int32Array, StringArray, TimestampMicrosecondArray, UInt64Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::execution::context::{SessionContext, TaskContext};
use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning as OutputPartitioning};
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties, collect,
};
use datafusion::prelude::SessionConfig;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, Literal, ManifestContentType, NestedField, PrimitiveLiteral, PrimitiveType,
    Schema, TableProperties, Transform, Type, UnboundPartitionSpec,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

const TARGET_PARTITIONS: usize = 8;
const SOURCE_PARTITIONS: usize = 4;
const FIXTURE_TARGET_FILE_SIZE_BYTES: usize = 1_048_576;

struct FailingExecutionPlan {
    properties: Arc<PlanProperties>,
}

impl FailingExecutionPlan {
    fn new(schema: Arc<ArrowSchema>) -> Self {
        Self {
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(schema),
                OutputPartitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
        }
    }
}

impl Debug for FailingExecutionPlan {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("FailingExecutionPlan")
    }
}

impl DisplayAs for FailingExecutionPlan {
    fn fmt_as(
        &self,
        _display_type: DisplayFormatType,
        formatter: &mut Formatter,
    ) -> std::fmt::Result {
        formatter.write_str("FailingExecutionPlan")
    }
}

impl ExecutionPlan for FailingExecutionPlan {
    fn name(&self) -> &str {
        "FailingExecutionPlan"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Err(DataFusionError::Execution(
            "controlled insert source failure".to_string(),
        ))
    }
}

fn source_batch(
    schema: Arc<ArrowSchema>,
    source_partition: usize,
    value_offset: usize,
) -> RecordBatch {
    let partition_values = [
        None,
        Some(0),
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
        Some(6),
    ];
    let range = value_offset..value_offset + 4;
    let ids = range
        .clone()
        .map(|index| i32::try_from(source_partition * 8 + index).expect("test id fits i32"))
        .collect::<Vec<_>>();
    let parts = range
        .clone()
        .map(|index| partition_values[index])
        .collect::<Vec<_>>();
    let labels = range
        .map(|index| format!("source-{source_partition}-value-{index}"))
        .collect::<Vec<_>>();

    RecordBatch::try_new(schema, vec![
        Arc::new(Int32Array::from(ids)),
        Arc::new(Int32Array::from(parts)),
        Arc::new(StringArray::from(labels)),
    ])
    .expect("build source batch")
}

async fn create_fixture(
    fanout_enabled: bool,
    target_partitions: usize,
) -> Result<(SessionContext, Arc<MemoryCatalog>, TableIdent, TempDir)> {
    let warehouse = TempDir::new().expect("create warehouse");
    let warehouse_path = warehouse
        .path()
        .to_str()
        .expect("warehouse path is UTF-8")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await?;
    let namespace = NamespaceIdent::new("insert_distribution".to_string());
    catalog.create_namespace(&namespace, HashMap::new()).await?;
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "part", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(3, "label", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "part", Transform::Identity)?
        .build();
    let table_ident = TableIdent::new(namespace.clone(), "target".to_string());
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("target".to_string())
                .location(format!("{warehouse_path}/target"))
                .schema(schema)
                .partition_spec(partition_spec)
                .properties(HashMap::from([
                    (
                        TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED.to_string(),
                        fanout_enabled.to_string(),
                    ),
                    (
                        TableProperties::PROPERTY_WRITE_TARGET_FILE_SIZE_BYTES.to_string(),
                        FIXTURE_TARGET_FILE_SIZE_BYTES.to_string(),
                    ),
                ]))
                .build(),
        )
        .await?;
    let catalog = Arc::new(catalog);
    let provider = Arc::new(IcebergCatalogProvider::try_new(catalog.clone()).await?);
    let config = SessionConfig::new().with_target_partitions(target_partitions);
    let context = SessionContext::new_with_config(config);
    context.register_catalog("catalog", provider);

    let source_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("part", DataType::Int32, true),
        Field::new("label", DataType::Utf8, false),
    ]));
    let source = (0..SOURCE_PARTITIONS)
        .map(|source_partition| {
            vec![
                source_batch(source_schema.clone(), source_partition, 0),
                source_batch(source_schema.clone(), source_partition, 4),
            ]
        })
        .collect::<Vec<_>>();
    let source = MemTable::try_new(source_schema, source).expect("build source table");
    context
        .register_table("source", Arc::new(source))
        .expect("register source table");

    Ok((context, catalog, table_ident, warehouse))
}

fn has_hash_repartition(plan: &Arc<dyn ExecutionPlan>, partition_count: usize) -> bool {
    if let Some(repartition) = plan.downcast_ref::<RepartitionExec>()
        && matches!(
            repartition.partitioning(),
            Partitioning::Hash(_, count) if *count == partition_count
        )
    {
        return true;
    }
    plan.children()
        .into_iter()
        .any(|child| has_hash_repartition(child, partition_count))
}

fn writer_input_is_sort(plan: &Arc<dyn ExecutionPlan>) -> Option<bool> {
    if plan.name() == "IcebergWriteExec" {
        return plan
            .children()
            .first()
            .map(|child| child.name() == "SortExec");
    }
    plan.children().into_iter().find_map(writer_input_is_sort)
}

fn find_write_exec(plan: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    if plan.name() == "IcebergWriteExec" {
        return Some(plan);
    }
    plan.children().into_iter().find_map(find_write_exec)
}

fn assert_writer_requirements(plan: &Arc<dyn ExecutionPlan>, fanout_enabled: bool) -> Result<()> {
    let write_exec = find_write_exec(plan).expect("optimized plan contains IcebergWriteExec");
    assert!(matches!(
        write_exec.required_input_distribution().as_slice(),
        [datafusion::physical_plan::Distribution::HashPartitioned(expressions)]
            if expressions.len() == 1
    ));
    assert_eq!(
        write_exec.required_input_ordering()[0].is_some(),
        !fanout_enabled
    );
    let replacement = Arc::new(EmptyExec::new(write_exec.children()[0].schema()));
    let rebuilt = Arc::clone(write_exec).with_new_children(vec![replacement])?;
    assert!(matches!(
        rebuilt.required_input_distribution().as_slice(),
        [datafusion::physical_plan::Distribution::HashPartitioned(expressions)]
            if expressions.len() == 1
    ));
    assert_eq!(
        rebuilt.required_input_ordering()[0].is_some(),
        !fanout_enabled
    );
    Ok(())
}

async fn live_file_census(
    catalog: &Arc<MemoryCatalog>,
    table_ident: &TableIdent,
) -> Result<BTreeMap<Option<i32>, Vec<u64>>> {
    let table = catalog.load_table(table_ident).await?;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("insert commits one snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await?;
    let mut census = BTreeMap::<Option<i32>, Vec<u64>>::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if !entry.is_alive() || entry.data_file().content_type() != DataContentType::Data {
                continue;
            }
            let value = match entry.data_file().partition().fields() {
                [Some(Literal::Primitive(PrimitiveLiteral::Int(value)))] => Some(*value),
                [None] => None,
                other => panic!("identity-int partition expected, got {other:?}"),
            };
            census
                .entry(value)
                .or_default()
                .push(entry.data_file().record_count());
        }
    }
    Ok(census)
}

fn transformed_source_batch(
    schema: Arc<ArrowSchema>,
    source_partition: usize,
    timestamp: i64,
) -> RecordBatch {
    RecordBatch::try_new(schema, vec![
        Arc::new(Int32Array::from(vec![0, 3])),
        Arc::new(TimestampMicrosecondArray::from(vec![timestamp, timestamp])),
        Arc::new(StringArray::from(vec![
            format!("source-{source_partition}-id-0-time-{timestamp}"),
            format!("source-{source_partition}-id-3-time-{timestamp}"),
        ])),
    ])
    .expect("build transformed source batch")
}

async fn create_transformed_fixture()
-> Result<(SessionContext, Arc<MemoryCatalog>, TableIdent, TempDir)> {
    let warehouse = TempDir::new().expect("create warehouse");
    let warehouse_path = warehouse
        .path()
        .to_str()
        .expect("warehouse path is UTF-8")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await?;
    let namespace = NamespaceIdent::new("transformed_distribution".to_string());
    catalog.create_namespace(&namespace, HashMap::new()).await?;
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "event_time", Type::Primitive(PrimitiveType::Timestamp))
                .into(),
            NestedField::required(3, "label", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(1, "id_bucket", Transform::Bucket(4))?
        .add_partition_field(2, "event_day", Transform::Day)?
        .build();
    let table_ident = TableIdent::new(namespace.clone(), "target".to_string());
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("target".to_string())
                .location(format!("{warehouse_path}/target"))
                .schema(schema)
                .partition_spec(partition_spec)
                .properties(HashMap::from([(
                    TableProperties::PROPERTY_WRITE_TARGET_FILE_SIZE_BYTES.to_string(),
                    FIXTURE_TARGET_FILE_SIZE_BYTES.to_string(),
                )]))
                .build(),
        )
        .await?;
    let catalog = Arc::new(catalog);
    let provider = Arc::new(IcebergCatalogProvider::try_new(catalog.clone()).await?);
    let context = SessionContext::new_with_config(
        SessionConfig::new().with_target_partitions(TARGET_PARTITIONS),
    );
    context.register_catalog("catalog", provider);

    let source_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("label", DataType::Utf8, false),
    ]));
    let day = 86_400_000_000;
    let source = (0..SOURCE_PARTITIONS)
        .map(|source_partition| {
            vec![
                transformed_source_batch(source_schema.clone(), source_partition, 0),
                transformed_source_batch(source_schema.clone(), source_partition, day),
            ]
        })
        .collect::<Vec<_>>();
    let source = MemTable::try_new(source_schema, source).expect("build transformed source table");
    context
        .register_table("source", Arc::new(source))
        .expect("register transformed source table");

    Ok((context, catalog, table_ident, warehouse))
}

async fn transformed_file_census(
    catalog: &Arc<MemoryCatalog>,
    table_ident: &TableIdent,
) -> Result<BTreeMap<String, Vec<u64>>> {
    let table = catalog.load_table(table_ident).await?;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("insert commits one snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await?;
    let mut census = BTreeMap::<String, Vec<u64>>::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.is_alive() && entry.data_file().content_type() == DataContentType::Data {
                census
                    .entry(format!("{:?}", entry.data_file().partition().fields()))
                    .or_default()
                    .push(entry.data_file().record_count());
            }
        }
    }
    Ok(census)
}

async fn assert_identity_rows(context: &SessionContext) -> Result<()> {
    let batches = context
        .sql(
            "SELECT id, part, label FROM catalog.insert_distribution.target \
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;
    assert_eq!(batches[0].schema().field(0).data_type(), &DataType::Int32);
    assert_eq!(batches[0].schema().field(1).data_type(), &DataType::Int32);
    assert_eq!(batches[0].schema().field(2).data_type(), &DataType::Utf8);
    let mut actual = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id is Int32");
        let parts = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("part is Int32");
        let labels = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("label is Utf8");
        for row in 0..batch.num_rows() {
            actual.push((
                ids.value(row),
                (!parts.is_null(row)).then(|| parts.value(row)),
                labels.value(row).to_string(),
            ));
        }
    }
    let partition_values = [
        None,
        Some(0),
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
        Some(6),
    ];
    let expected = (0..SOURCE_PARTITIONS)
        .flat_map(|source_partition| {
            partition_values
                .into_iter()
                .enumerate()
                .map(move |(index, part)| {
                    (
                        i32::try_from(source_partition * 8 + index).expect("test id fits i32"),
                        part,
                        format!("source-{source_partition}-value-{index}"),
                    )
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    Ok(())
}

async fn assert_identity_insert(fanout_enabled: bool, target_partitions: usize) -> Result<()> {
    let (context, catalog, table_ident, _warehouse) =
        create_fixture(fanout_enabled, target_partitions).await?;
    let dataframe = context
        .sql(
            "INSERT INTO catalog.insert_distribution.target \
             SELECT id, part, label FROM source",
        )
        .await?;
    let plan = dataframe.create_physical_plan().await?;
    let has_hash_repartition = has_hash_repartition(&plan, target_partitions);
    let writer_input_is_sort = writer_input_is_sort(&plan);
    assert_writer_requirements(&plan, fanout_enabled)?;
    let result = collect(plan, context.task_ctx()).await?;
    let inserted = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("insert result is UInt64")
        .value(0);
    assert_eq!(inserted, 32);

    let rows = context
        .sql(
            "SELECT count(*) AS rows, count(part) AS non_null_parts, \
             sum(id) AS id_sum FROM catalog.insert_distribution.target",
        )
        .await?
        .collect()
        .await?;
    assert_eq!(rows[0].schema().field(0).data_type(), &DataType::Int64);
    assert_eq!(rows[0].schema().field(1).data_type(), &DataType::Int64);
    assert_eq!(rows[0].schema().field(2).data_type(), &DataType::Int64);
    let values = datafusion::arrow::util::pretty::pretty_format_batches(&rows)
        .expect("format aggregate rows")
        .to_string();
    assert!(
        values.contains("| 32   | 28             | 496    |"),
        "{values}"
    );
    assert_identity_rows(&context).await?;

    let census = live_file_census(&catalog, &table_ident).await?;
    assert_eq!(census.len(), TARGET_PARTITIONS);
    assert!(
        census.values().all(|record_counts| record_counts == &[4]),
        "controlled fixture expects one four-row file per value: {census:?}"
    );
    if target_partitions > 1 {
        assert!(
            has_hash_repartition,
            "optimized INSERT plan removed the target hash exchange; census={census:?}"
        );
    }
    assert_eq!(writer_input_is_sort, Some(!fanout_enabled));
    Ok(())
}

#[tokio::test]
async fn fanout_insert_hashes_each_partition_value_to_one_writer() -> Result<()> {
    assert_identity_insert(true, TARGET_PARTITIONS).await
}

#[tokio::test]
async fn clustered_insert_hashes_then_sorts_each_writer_input() -> Result<()> {
    assert_identity_insert(false, TARGET_PARTITIONS).await
}

#[tokio::test]
async fn single_target_partition_preserves_rows_and_partition_files() -> Result<()> {
    assert_identity_insert(true, 1).await
}

#[tokio::test]
async fn unpartitioned_insert_keeps_unspecified_writer_distribution() -> Result<()> {
    let warehouse = TempDir::new().expect("create warehouse");
    let warehouse_path = warehouse
        .path()
        .to_str()
        .expect("warehouse path is UTF-8")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await?;
    let namespace = NamespaceIdent::new("unpartitioned_distribution".to_string());
    catalog.create_namespace(&namespace, HashMap::new()).await?;
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "label", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("target".to_string())
                .location(format!("{warehouse_path}/target"))
                .schema(schema)
                .properties(HashMap::new())
                .build(),
        )
        .await?;
    let provider = Arc::new(IcebergCatalogProvider::try_new(Arc::new(catalog)).await?);
    let context = SessionContext::new_with_config(
        SessionConfig::new().with_target_partitions(TARGET_PARTITIONS),
    );
    context.register_catalog("catalog", provider);
    let source_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let source = (0..SOURCE_PARTITIONS)
        .map(|source_partition| {
            vec![
                RecordBatch::try_new(source_schema.clone(), vec![
                    Arc::new(Int32Array::from(vec![
                        i32::try_from(source_partition * 2).expect("test id fits i32"),
                        i32::try_from(source_partition * 2 + 1).expect("test id fits i32"),
                    ])),
                    Arc::new(StringArray::from(vec![
                        format!("source-{source_partition}-0"),
                        format!("source-{source_partition}-1"),
                    ])),
                ])
                .expect("build unpartitioned source batch"),
            ]
        })
        .collect::<Vec<_>>();
    context.register_table(
        "source",
        Arc::new(MemTable::try_new(source_schema, source)?),
    )?;

    let dataframe = context
        .sql(
            "INSERT INTO catalog.unpartitioned_distribution.target \
             SELECT id, label FROM source",
        )
        .await?;
    let plan = dataframe.create_physical_plan().await?;
    assert!(!has_hash_repartition(&plan, TARGET_PARTITIONS));
    let write_exec = find_write_exec(&plan).expect("optimized plan contains IcebergWriteExec");
    assert!(matches!(
        write_exec.required_input_distribution().as_slice(),
        [datafusion::physical_plan::Distribution::UnspecifiedDistribution]
    ));
    assert!(write_exec.required_input_ordering()[0].is_none());
    let result = collect(plan, context.task_ctx()).await?;
    let inserted = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("insert result is UInt64")
        .value(0);
    assert_eq!(inserted, 8);
    let rows = context
        .sql(
            "SELECT count(*) AS rows, sum(id) AS id_sum \
             FROM catalog.unpartitioned_distribution.target",
        )
        .await?
        .collect()
        .await?;
    let values = datafusion::arrow::util::pretty::pretty_format_batches(&rows)
        .expect("format unpartitioned aggregate rows")
        .to_string();
    assert!(values.contains("| 8    | 28     |"), "{values}");
    Ok(())
}

#[tokio::test]
async fn zero_row_partitioned_insert_commits_an_empty_snapshot() -> Result<()> {
    let (context, catalog, table_ident, _warehouse) =
        create_fixture(true, TARGET_PARTITIONS).await?;
    let dataframe = context
        .sql(
            "INSERT INTO catalog.insert_distribution.target \
             SELECT id, part, label FROM source WHERE false",
        )
        .await?;
    let plan = dataframe.create_physical_plan().await?;
    assert_writer_requirements(&plan, true)?;
    let result = collect(plan, context.task_ctx()).await?;
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].num_rows(), 1);
    let inserted = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("insert result is UInt64")
        .value(0);
    assert_eq!(inserted, 0);

    let table = catalog.load_table(&table_ident).await?;
    assert!(table.metadata().current_snapshot().is_some());
    assert!(live_file_census(&catalog, &table_ident).await?.is_empty());
    let rows = context
        .sql("SELECT count(*) AS rows FROM catalog.insert_distribution.target")
        .await?
        .collect()
        .await?;
    assert_eq!(rows[0].schema().field(0).data_type(), &DataType::Int64);
    let values = datafusion::arrow::util::pretty::pretty_format_batches(&rows)
        .expect("format zero-row aggregate")
        .to_string();
    assert!(values.contains("| 0    |"), "{values}");
    Ok(())
}

#[tokio::test]
async fn source_failure_propagates_without_committing_a_snapshot() -> Result<()> {
    let (context, catalog, table_ident, _warehouse) =
        create_fixture(true, TARGET_PARTITIONS).await?;
    let provider = context
        .table_provider("catalog.insert_distribution.target")
        .await?;
    let input = Arc::new(FailingExecutionPlan::new(provider.schema()));
    let plan = provider
        .insert_into(&context.state(), input, InsertOp::Append)
        .await?;
    let error = collect(plan, context.task_ctx())
        .await
        .expect_err("source error must fail the insert");
    assert!(
        error
            .to_string()
            .contains("controlled insert source failure")
    );
    let table = catalog.load_table(&table_ident).await?;
    assert!(table.metadata().current_snapshot().is_none());
    Ok(())
}

#[tokio::test]
async fn bucket_and_day_values_co_locate_across_source_tasks() -> Result<()> {
    let (context, catalog, table_ident, _warehouse) = create_transformed_fixture().await?;
    let dataframe = context
        .sql(
            "INSERT INTO catalog.transformed_distribution.target \
             SELECT id, event_time, label FROM source",
        )
        .await?;
    let plan = dataframe.create_physical_plan().await?;
    assert!(has_hash_repartition(&plan, TARGET_PARTITIONS));
    let result = collect(plan, context.task_ctx()).await?;
    let inserted = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("insert result is UInt64")
        .value(0);
    assert_eq!(inserted, 16);

    let rows = context
        .sql(
            "SELECT count(*) AS rows, sum(id) AS id_sum \
             FROM catalog.transformed_distribution.target",
        )
        .await?
        .collect()
        .await?;
    let values = datafusion::arrow::util::pretty::pretty_format_batches(&rows)
        .expect("format transformed aggregate rows")
        .to_string();
    assert!(values.contains("| 16   | 24     |"), "{values}");

    let census = transformed_file_census(&catalog, &table_ident).await?;
    assert_eq!(
        census.len(),
        4,
        "expected two bucket values across two day values: {census:?}"
    );
    assert!(
        census.values().all(|record_counts| record_counts == &[4]),
        "equal bucket-and-day values must share one writer: {census:?}"
    );
    Ok(())
}
