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

#![allow(dead_code)]

use std::collections::HashMap;
use std::ops::Not;
use std::sync::Arc;

use anyhow::Result;
use datafusion::arrow::array::{
    Array, BooleanArray, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeBinaryArray, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionConfig;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, FormatVersion, ManifestContentType, NestedField, NullOrder, PrimitiveType,
    Schema, SortDirection, SortField, TableProperties, Transform, Type, UnboundPartitionSpec,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

pub fn sort_field(
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

pub struct Fixture {
    pub context: SessionContext,
    pub catalog: Arc<MemoryCatalog>,
    pub ident: TableIdent,
    pub namespace: String,
    pub table: String,
    pub _warehouse: TempDir,
}

pub async fn fixture_with_props(
    namespace: &str,
    table: &str,
    schema: Schema,
    partition_spec: UnboundPartitionSpec,
    sort_order: Option<iceberg::spec::SortOrder>,
    target_partitions: usize,
    properties: HashMap<String, String>,
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
        .properties(properties);
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

pub async fn fixture(
    namespace: &str,
    table: &str,
    schema: Schema,
    partition_spec: UnboundPartitionSpec,
    sort_order: Option<iceberg::spec::SortOrder>,
    target_partitions: usize,
) -> Result<Fixture> {
    fixture_with_props(
        namespace,
        table,
        schema,
        partition_spec,
        sort_order,
        target_partitions,
        HashMap::from([(
            TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED.to_string(),
            "true".to_string(),
        )]),
    )
    .await
}

pub async fn run_insert(fixture: &Fixture, source: MemTable, select: &str) -> Result<()> {
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

pub async fn run_insert_overwrite(fixture: &Fixture, source: MemTable, select: &str) -> Result<()> {
    let _ = fixture.context.deregister_table("source");
    fixture
        .context
        .register_table("source", Arc::new(source))
        .expect("register source");
    fixture
        .context
        .sql(&format!(
            "INSERT OVERWRITE catalog.{}.{} {select}",
            fixture.namespace, fixture.table
        ))
        .await?
        .collect()
        .await?;
    Ok(())
}

pub fn local_path(file_path: &str) -> &str {
    file_path.strip_prefix("file://").unwrap_or(file_path)
}

pub async fn live_files(fixture: &Fixture) -> Result<Vec<(String, Option<i32>)>> {
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

pub async fn live_row_count(fixture: &Fixture) -> Result<usize> {
    let files = live_files(fixture).await?;
    let mut rows = 0;
    for (path, _) in &files {
        let file = std::fs::File::open(local_path(path)).expect("open data file");
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("parquet reader")
            .build()
            .expect("build reader");
        for batch in reader {
            rows += batch.expect("read batch").num_rows();
        }
    }
    Ok(rows)
}

pub async fn default_order_id(fixture: &Fixture) -> Result<i32> {
    let table = fixture.catalog.load_table(&fixture.ident).await?;
    i32::try_from(table.metadata().default_sort_order_id()).map_err(anyhow::Error::from)
}

pub fn read_long_column(path: &str, index: usize) -> Vec<i64> {
    read_nullable_long_column(path, index)
        .into_iter()
        .map(|value| {
            assert!(value.is_some(), "expected non-null id in {path}");
            value.expect("non-null id")
        })
        .collect()
}

pub fn writer_input_is_sort(plan: &Arc<dyn ExecutionPlan>) -> Option<bool> {
    if plan.name() == "IcebergWriteExec" {
        return plan
            .children()
            .first()
            .map(|child| child.name() == "SortExec");
    }
    plan.children().into_iter().find_map(writer_input_is_sort)
}

pub fn unpartitioned_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder().with_spec_id(0).build()
}

pub fn id_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "p", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("id schema")
}

pub fn id_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("p", DataType::Int32, false),
    ]))
}

pub fn shuffled_ids(count: i64) -> Vec<i64> {
    (0..count).map(|index| (index * 7919) % count).collect()
}

pub fn id_batches(ids: &[i64]) -> Vec<RecordBatch> {
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

pub fn read_int_column(path: &str, index: usize) -> Vec<Option<i32>> {
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

pub fn read_nullable_long_column(path: &str, index: usize) -> Vec<Option<i64>> {
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

pub fn read_float_column(path: &str, index: usize) -> Vec<Option<f32>> {
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
            .downcast_ref::<Float32Array>()
            .expect("float column");
        for row in 0..batch.num_rows() {
            values.push(column.is_null(row).not().then(|| column.value(row)));
        }
    }
    values
}

pub fn read_double_column(path: &str, index: usize) -> Vec<Option<f64>> {
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
            .downcast_ref::<Float64Array>()
            .expect("double column");
        for row in 0..batch.num_rows() {
            values.push(column.is_null(row).not().then(|| column.value(row)));
        }
    }
    values
}

pub fn read_string_column(path: &str, index: usize) -> Vec<Option<String>> {
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
            .downcast_ref::<StringArray>()
            .expect("string column");
        for row in 0..batch.num_rows() {
            values.push(
                column
                    .is_null(row)
                    .not()
                    .then(|| column.value(row).to_string()),
            );
        }
    }
    values
}

pub fn read_decimal_column(path: &str, index: usize) -> Vec<Option<i128>> {
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
            .downcast_ref::<Decimal128Array>()
            .expect("decimal column");
        for row in 0..batch.num_rows() {
            values.push(column.is_null(row).not().then(|| column.value(row)));
        }
    }
    values
}

pub fn read_boolean_column(path: &str, index: usize) -> Vec<Option<bool>> {
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
            .downcast_ref::<BooleanArray>()
            .expect("boolean column");
        for row in 0..batch.num_rows() {
            values.push(column.is_null(row).not().then(|| column.value(row)));
        }
    }
    values
}

pub fn read_binary_column(path: &str, index: usize) -> Vec<Option<Vec<u8>>> {
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
            .downcast_ref::<LargeBinaryArray>()
            .expect("binary column");
        for row in 0..batch.num_rows() {
            values.push(
                column
                    .is_null(row)
                    .not()
                    .then(|| column.value(row).to_vec()),
            );
        }
    }
    values
}

pub fn read_timestamp_column(path: &str, index: usize) -> Vec<Option<i64>> {
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
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("timestamp column");
        for row in 0..batch.num_rows() {
            values.push(column.is_null(row).not().then(|| column.value(row)));
        }
    }
    values
}
