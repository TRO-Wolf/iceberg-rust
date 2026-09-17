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

use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, TableProperties, Type};
use iceberg::table::Table;
use iceberg::transaction::StagedTableTransaction;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use tempfile::TempDir;

fn test_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("schema builds")
}

async fn create_with_properties(
    properties: HashMap<String, String>,
) -> iceberg::Result<(Table, TempDir)> {
    let warehouse = TempDir::new().expect("create warehouse");
    let warehouse_path = warehouse
        .path()
        .to_str()
        .expect("warehouse path is UTF-8")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(std::sync::Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await?;
    let namespace = NamespaceIdent::new("create_defaults".to_string());
    catalog.create_namespace(&namespace, HashMap::new()).await?;
    let table = catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("t".to_string())
                .location(format!("{warehouse_path}/t"))
                .schema(test_schema())
                .properties(properties)
                .build(),
        )
        .await?;
    Ok((table, warehouse))
}

#[tokio::test]
async fn create_table_stamps_zstd_codec_by_default() -> iceberg::Result<()> {
    let (table, _warehouse) = create_with_properties(HashMap::new()).await?;
    assert_eq!(
        table
            .metadata()
            .properties()
            .get(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC),
        Some(&TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC_DEFAULT.to_string())
    );
    Ok(())
}

#[tokio::test]
async fn create_table_keeps_explicit_codec() -> iceberg::Result<()> {
    let (table, _warehouse) = create_with_properties(HashMap::from([(
        TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC.to_string(),
        "snappy".to_string(),
    )]))
    .await?;
    assert_eq!(
        table
            .metadata()
            .properties()
            .get(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC),
        Some(&"snappy".to_string())
    );
    Ok(())
}

#[tokio::test]
async fn replace_table_stamps_zstd_codec_by_default() -> iceberg::Result<()> {
    let (table, _warehouse) = create_with_properties(HashMap::new()).await?;
    let staged = StagedTableTransaction::begin_replace(
        &table,
        TableCreation::builder()
            .name(table.identifier().name().to_string())
            .schema(test_schema())
            .properties(HashMap::new())
            .build(),
    )
    .await?;
    assert_eq!(
        staged
            .table()
            .metadata()
            .properties()
            .get(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC),
        Some(&TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC_DEFAULT.to_string())
    );
    Ok(())
}
