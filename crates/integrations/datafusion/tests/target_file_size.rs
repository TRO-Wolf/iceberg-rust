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
use std::sync::Arc;

use anyhow::Result;
use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, ManifestContentType, NestedField, PrimitiveType, Schema, TableProperties, Type,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

const STREAMS: usize = 4;
const STREAM_ROWS: usize = 50_000;
const TARGET_262144: usize = 262_144;
const TARGET_1MB: usize = 1_048_576;

fn oracle_batch(partition: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let start = (partition * STREAM_ROWS) as i64;
    let ids: Vec<i64> = (start..start + STREAM_ROWS as i64).collect();
    let strings: Vec<String> = ids.iter().map(|id| format!("row-{}", id * 7919)).collect();
    RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(ids)),
        Arc::new(StringArray::from(strings)),
    ])
    .expect("oracle batch builds")
}

async fn create_oracle_table(
    properties: HashMap<String, String>,
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
    let namespace = NamespaceIdent::new("target_file_size".to_string());
    catalog.create_namespace(&namespace, HashMap::new()).await?;
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "s", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    let table_ident = TableIdent::new(namespace.clone(), "target".to_string());
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("target".to_string())
                .location(format!("{warehouse_path}/target"))
                .schema(schema)
                .properties(properties)
                .build(),
        )
        .await?;
    let catalog = Arc::new(catalog);
    let provider = Arc::new(IcebergCatalogProvider::try_new(catalog.clone()).await?);
    let context =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(STREAMS));
    context.register_catalog("catalog", provider);

    let source_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let source = (0..STREAMS)
        .map(|partition| vec![oracle_batch(partition)])
        .collect::<Vec<_>>();
    let source = MemTable::try_new(source_schema, source).expect("build source table");
    context
        .register_table("source", Arc::new(source))
        .expect("register source table");
    Ok((context, catalog, table_ident, warehouse))
}

async fn file_census(
    catalog: &Arc<MemoryCatalog>,
    table_ident: &TableIdent,
) -> Result<Vec<(u64, u64)>> {
    let table = catalog.load_table(table_ident).await?;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("insert commits one snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await?;
    let mut census = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if !entry.is_alive() || entry.data_file().content_type() != DataContentType::Data {
                continue;
            }
            census.push((
                entry.data_file().record_count(),
                entry.data_file().file_size_in_bytes(),
            ));
        }
    }
    Ok(census)
}

async fn insert_target_size(
    target: Option<usize>,
) -> Result<(Vec<(u64, u64)>, TableIdent, TempDir)> {
    let mut properties = HashMap::new();
    if let Some(target) = target {
        properties.insert(
            TableProperties::PROPERTY_WRITE_TARGET_FILE_SIZE_BYTES.to_string(),
            target.to_string(),
        );
    }
    let (context, catalog, table_ident, warehouse) = create_oracle_table(properties).await?;
    let codec = catalog
        .load_table(&table_ident)
        .await?
        .metadata()
        .properties()
        .get(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC)
        .cloned();
    assert_eq!(codec.as_deref(), Some("zstd"));
    context
        .sql("INSERT INTO catalog.target_file_size.target SELECT id, s FROM source")
        .await?
        .collect()
        .await?;
    let census = file_census(&catalog, &table_ident).await?;
    Ok((census, table_ident, warehouse))
}

#[tokio::test]
async fn insert_262144_matches_spark_file_count_and_max_size() -> Result<()> {
    let (census, _, _warehouse) = insert_target_size(Some(TARGET_262144)).await?;
    let total_records: u64 = census.iter().map(|(records, _)| records).sum();
    assert_eq!(total_records, 200_000);
    assert!(
        (15..=25).contains(&census.len()),
        "expected 15-25 files, got {}: {census:?}",
        census.len()
    );
    let max_size: u64 = census
        .iter()
        .map(|(_, bytes)| bytes)
        .max()
        .copied()
        .unwrap_or(0);
    assert!(
        max_size <= 71_473,
        "expected max file size <= 71,473, got {max_size}: {census:?}"
    );
    Ok(())
}

#[tokio::test]
async fn insert_1mb_keeps_one_file_per_stream() -> Result<()> {
    let (census, _, _warehouse) = insert_target_size(Some(TARGET_1MB)).await?;
    let total_records: u64 = census.iter().map(|(records, _)| records).sum();
    assert_eq!(total_records, 200_000);
    assert_eq!(census.len(), STREAMS);
    Ok(())
}

#[tokio::test]
async fn insert_default_target_keeps_one_file_per_stream() -> Result<()> {
    let (census, _, _warehouse) = insert_target_size(None).await?;
    let total_records: u64 = census.iter().map(|(records, _)| records).sum();
    assert_eq!(total_records, 200_000);
    assert_eq!(census.len(), STREAMS);
    Ok(())
}
