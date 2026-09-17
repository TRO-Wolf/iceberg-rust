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
use std::ops::Not;
use std::sync::Arc;

use anyhow::Result;
use datafusion::arrow::array::{Array, Int32Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionConfig;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, FormatVersion, ManifestContentType, NestedField, NullOrder, PrimitiveType,
    Schema, SortDirection, SortField, SortOrder, TableProperties, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::transform::create_transform_function;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

fn sort_field(
    source_id: i32,
    transform: Transform,
    direction: SortDirection,
    null_order: NullOrder,
) -> SortField {
    SortField::builder()
        .source_id(source_id)
        .transform(transform)
        .direction(direction)
        .null_order(null_order)
        .build()
}

fn id_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "p", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("id schema")
}

fn id_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("p", DataType::Int32, false),
    ]))
}

fn shuffled_ids(count: i64) -> Vec<i64> {
    (0..count).map(|index| (index * 7919) % count).collect()
}

fn id_batches(ids: &[i64]) -> Vec<RecordBatch> {
    let batch = RecordBatch::try_new(id_arrow_schema(), vec![
        Arc::new(Int64Array::from(ids.to_vec())),
        Arc::new(Int32Array::from(
            ids.iter()
                .map(|id| i32::try_from(id % 2).expect("partition value fits i32"))
                .collect::<Vec<_>>(),
        )),
    ])
    .expect("id batch");
    vec![batch]
}

struct Fixture {
    context: SessionContext,
    catalog: Arc<MemoryCatalog>,
    ident: TableIdent,
    namespace: String,
    table: String,
    _warehouse: TempDir,
}

async fn fixture(
    namespace: &str,
    table: &str,
    schema: Schema,
    partition_spec: UnboundPartitionSpec,
    sort_order: Option<SortOrder>,
    target_partitions: usize,
) -> Result<Fixture> {
    let warehouse = TempDir::new().expect("warehouse");
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
    let namespace_ident = NamespaceIdent::new(namespace.to_string());
    catalog
        .create_namespace(&namespace_ident, HashMap::new())
        .await?;
    let creation = TableCreation::builder()
        .name(table.to_string())
        .location(format!("{warehouse_path}/{table}"))
        .schema(schema)
        .partition_spec(partition_spec)
        .sort_order_opt(sort_order)
        .format_version(FormatVersion::V2)
        .properties(HashMap::from([(
            TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED.to_string(),
            "true".to_string(),
        )]));
    catalog
        .create_table(&namespace_ident, creation.build())
        .await?;
    let catalog = Arc::new(catalog);
    let provider = Arc::new(IcebergCatalogProvider::try_new(catalog.clone()).await?);
    let context = SessionContext::new_with_config(
        SessionConfig::new().with_target_partitions(target_partitions),
    );
    context.register_catalog("catalog", provider);
    Ok(Fixture {
        context,
        catalog,
        ident: TableIdent::new(namespace_ident, table.to_string()),
        namespace: namespace.to_string(),
        table: table.to_string(),
        _warehouse: warehouse,
    })
}

async fn run_insert(fixture: &Fixture, source: MemTable, select: &str) -> Result<()> {
    fixture
        .context
        .register_table("source", Arc::new(source))
        .expect("register source");
    fixture
        .context
        .sql(&format!(
            "INSERT INTO catalog.{}.{} {select}",
            fixture.namespace, fixture.table
        ))
        .await?
        .collect()
        .await?;
    Ok(())
}

fn local_path(file_path: &str) -> &str {
    file_path.strip_prefix("file://").unwrap_or(file_path)
}

async fn live_files(fixture: &Fixture) -> Result<Vec<(String, Option<i32>)>> {
    let table = fixture.catalog.load_table(&fixture.ident).await?;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("insert commits one snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await?;
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if !entry.is_alive() || entry.data_file().content_type() != DataContentType::Data {
                continue;
            }
            files.push((
                entry.data_file().file_path().to_string(),
                entry.data_file().sort_order_id(),
            ));
        }
    }
    Ok(files)
}

async fn default_order_id(fixture: &Fixture) -> Result<i32> {
    let table = fixture.catalog.load_table(&fixture.ident).await?;
    i32::try_from(table.metadata().default_sort_order_id()).map_err(anyhow::Error::from)
}

fn read_long_column(path: &str, index: usize) -> Vec<i64> {
    let file = std::fs::File::open(local_path(path)).expect("open data file");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("parquet reader")
        .build()
        .expect("build reader");
    let mut values = Vec::new();
    for batch in reader {
        let batch = batch.expect("read batch");
        let column = batch
            .column(index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("long column");
        for row in 0..batch.num_rows() {
            assert!(!column.is_null(row), "expected non-null id in {path}");
            values.push(column.value(row));
        }
    }
    values
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

fn unpartitioned_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder().with_spec_id(0).build()
}

fn nulls_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "b", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("nulls schema")
}

fn nulls_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int64, true),
    ]))
}

fn read_int_column(path: &str, index: usize) -> Vec<Option<i32>> {
    let file = std::fs::File::open(local_path(path)).expect("open data file");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("parquet reader")
        .build()
        .expect("build reader");
    let mut values = Vec::new();
    for batch in reader {
        let batch = batch.expect("read batch");
        let column = batch
            .column(index)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("int column");
        for row in 0..batch.num_rows() {
            values.push(column.is_null(row).not().then(|| column.value(row)));
        }
    }
    values
}

fn read_nullable_long_column(path: &str, index: usize) -> Vec<Option<i64>> {
    let file = std::fs::File::open(local_path(path)).expect("open data file");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("parquet reader")
        .build()
        .expect("build reader");
    let mut values = Vec::new();
    for batch in reader {
        let batch = batch.expect("read batch");
        let column = batch
            .column(index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("long column");
        for row in 0..batch.num_rows() {
            values.push(column.is_null(row).not().then(|| column.value(row)));
        }
    }
    values
}

#[tokio::test]
async fn insert_into_table_with_asc_order_writes_one_sorted_file_with_order_stamp() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let fixture = fixture(
        "sorted_asc",
        "t",
        id_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let ids = shuffled_ids(1000);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let rows = read_long_column(&files[0].0, 0);
    assert_eq!(rows.len(), 1000);
    let mut expected = rows.clone();
    expected.sort_unstable();
    assert_eq!(rows, expected, "file rows are ascending by id");
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_table_with_desc_order_writes_descending_file() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Descending,
            NullOrder::Last,
        ))
        .build(&id_schema())?;
    let fixture = fixture(
        "sorted_desc",
        "t",
        id_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let ids = shuffled_ids(1000);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let rows = read_long_column(&files[0].0, 0);
    assert_eq!(rows.len(), 1000);
    let mut expected = rows.clone();
    expected.sort_unstable_by(|left, right| right.cmp(left));
    assert_eq!(rows, expected, "file rows are descending by id");
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_unsorted_table_adds_no_sort_and_stamps_zero() -> Result<()> {
    let fixture = fixture(
        "sorted_none",
        "t",
        id_schema(),
        unpartitioned_spec(),
        None,
        1,
    )
    .await?;
    let plan = fixture
        .context
        .sql("INSERT INTO catalog.sorted_none.t SELECT CAST(1 AS BIGINT) AS id, 1 AS p")
        .await?
        .create_physical_plan()
        .await?;
    assert_eq!(
        writer_input_is_sort(&plan),
        Some(false),
        "no SortExec feeds the write when the table has no sort order"
    );
    fixture
        .context
        .sql("INSERT INTO catalog.sorted_none.t SELECT CAST(1 AS BIGINT) AS id, 1 AS p")
        .await?
        .collect()
        .await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1);
    for (_, stamp) in &files {
        assert_eq!(*stamp, Some(0), "unsorted table stamps order id 0");
    }
    Ok(())
}

#[tokio::test]
async fn insert_into_sorted_table_feeds_write_through_sort() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let fixture = fixture(
        "sorted_plan",
        "t",
        id_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let ids = shuffled_ids(100);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)]).expect("plan source");
    fixture
        .context
        .register_table("source", Arc::new(source))
        .expect("register plan source");
    let plan = fixture
        .context
        .sql("INSERT INTO catalog.sorted_plan.t SELECT id, p FROM source")
        .await?
        .create_physical_plan()
        .await?;
    assert_eq!(
        writer_input_is_sort(&plan),
        Some(true),
        "a SortExec feeds the write when the table declares a sort order"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_two_key_order_honours_direction_and_null_order() -> Result<()> {
    let schema = nulls_schema();
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::Last,
        ))
        .with_sort_field(sort_field(
            2,
            Transform::Identity,
            SortDirection::Descending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let fixture = fixture(
        "sorted_nulls",
        "t",
        schema,
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let rows: Vec<(Option<i32>, Option<i64>)> = vec![
        (Some(2), Some(1)),
        (None, Some(5)),
        (Some(1), None),
        (Some(0), Some(7)),
        (None, None),
        (Some(1), Some(3)),
        (Some(2), None),
        (Some(1), Some(1)),
    ];
    let batch = RecordBatch::try_new(nulls_arrow_schema(), vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|row| row.0).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.1).collect::<Vec<_>>(),
        )),
    ])
    .expect("nulls batch");
    let source = MemTable::try_new(nulls_arrow_schema(), vec![vec![batch]])?;
    run_insert(&fixture, source, "SELECT a, b FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let actual = read_int_column(&files[0].0, 0)
        .into_iter()
        .zip(read_nullable_long_column(&files[0].0, 1))
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        vec![
            (Some(0), Some(7)),
            (Some(1), None),
            (Some(1), Some(3)),
            (Some(1), Some(1)),
            (Some(2), None),
            (Some(2), Some(1)),
            (None, None),
            (None, Some(5)),
        ],
        "file rows follow a ASC NULLS LAST, b DESC NULLS FIRST"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_partitioned_table_sorts_within_each_file() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "p", Transform::Identity)?
        .build();
    let fixture = fixture("sorted_part", "t", id_schema(), spec, Some(order), 1).await?;
    let ids = shuffled_ids(1000);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 2, "one file per partition value");
    let expected_id = default_order_id(&fixture).await?;
    for (path, stamp) in &files {
        assert_eq!(*stamp, Some(expected_id), "data file stamps the order id");
        let rows = read_long_column(path, 0);
        assert_eq!(rows.len(), 500);
        let mut expected = rows.clone();
        expected.sort_unstable();
        assert_eq!(rows, expected, "file {path} rows are ascending by id");
        let parts = read_int_column(path, 1);
        assert!(
            parts.iter().all(|part| *part == parts[0]),
            "file {path} holds a single partition value"
        );
    }
    Ok(())
}

#[tokio::test]
async fn insert_into_partitioned_table_sorts_every_file_across_streams() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "p", Transform::Identity)?
        .build();
    let fixture = fixture("sorted_streams", "t", id_schema(), spec, Some(order), 4).await?;
    let ids = shuffled_ids(1000);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert!(!files.is_empty(), "insert writes data files");
    let expected_id = default_order_id(&fixture).await?;
    for (path, stamp) in &files {
        assert_eq!(*stamp, Some(expected_id), "data file stamps the order id");
        let rows = read_long_column(path, 0);
        let mut expected = rows.clone();
        expected.sort_unstable();
        assert_eq!(rows, expected, "file {path} rows are ascending by id");
    }
    Ok(())
}

#[tokio::test]
async fn insert_into_bucket_order_sorts_by_bucket_then_id() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Bucket(4),
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let fixture = fixture(
        "sorted_bucket",
        "t",
        id_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let ids = shuffled_ids(1000);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let rows = read_long_column(&files[0].0, 0);
    assert_eq!(rows.len(), 1000);
    let buckets = create_transform_function(&Transform::Bucket(4))?
        .transform(Arc::new(Int64Array::from(rows.clone())))?;
    let buckets = buckets
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("bucket values are Int32");
    let mut previous = (i32::MIN, i64::MIN);
    for (row, id) in rows.iter().enumerate() {
        let key = (buckets.value(row), *id);
        assert!(
            key >= previous,
            "row {row} breaks (bucket, id) order: {key:?} after {previous:?}"
        );
        previous = key;
    }
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}
