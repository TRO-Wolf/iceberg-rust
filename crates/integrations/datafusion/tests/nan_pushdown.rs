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

use datafusion::arrow::array::{Array, Int32Array};
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::{Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, Result, TableCreation};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

async fn get_iceberg_catalog() -> MemoryCatalog {
    let temp_dir = TempDir::new().unwrap();
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                temp_dir.path().to_str().unwrap().to_string(),
            )]),
        )
        .await
        .unwrap()
}

async fn select_ids(ctx: &SessionContext, table: &str, filter: &str) -> Vec<i32> {
    let batches = ctx
        .sql(&format!("SELECT id FROM {table} WHERE {filter}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut ids: Vec<i32> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id column is Int32")
                .iter()
                .map(|value| value.expect("id is a required column"))
                .collect::<Vec<_>>()
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn nan_comparison_filters_return_spark_rows() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_nan_pushdown".to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "d", Type::Primitive(PrimitiveType::Double)).into(),
            NestedField::optional(3, "f", Type::Primitive(PrimitiveType::Float)).into(),
        ])
        .build()?;
    let temp_dir = TempDir::new().unwrap();
    let creation = TableCreation::builder()
        .location(temp_dir.path().to_str().unwrap().to_string())
        .name("my_table".to_string())
        .properties(HashMap::new())
        .schema(schema)
        .build();
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);
    let table = "catalog.test_nan_pushdown.my_table";

    ctx.sql(&format!(
        "INSERT INTO {table} VALUES (1, CAST('NaN' AS DOUBLE), CAST('NaN' AS FLOAT)), (2, 1.0, CAST(1.5 AS FLOAT))"
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx.sql(&format!(
        "INSERT INTO {table} VALUES (3, NULL, NULL), (4, 2.0, CAST('NaN' AS FLOAT)), (5, CAST('NaN' AS DOUBLE), CAST(3.0 AS FLOAT))"
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    for (filter, expected_ids) in [
        ("d = CAST('NaN' AS DOUBLE)", vec![1, 5]),
        ("d IN (CAST('NaN' AS DOUBLE))", vec![1, 5]),
        ("d IN (CAST('NaN' AS DOUBLE), 1.0)", vec![1, 2, 5]),
        ("d < CAST('NaN' AS DOUBLE)", vec![2, 4]),
        ("d >= CAST('NaN' AS DOUBLE)", vec![1, 5]),
        ("d != CAST('NaN' AS DOUBLE)", vec![2, 4]),
        ("d <=> CAST('NaN' AS DOUBLE)", vec![1, 5]),
        ("f = CAST('NaN' AS FLOAT)", vec![1, 4]),
        ("f = CAST('NaN' AS DOUBLE)", vec![1, 4]),
        ("f IN (CAST('NaN' AS FLOAT), CAST(1.5 AS FLOAT))", vec![
            1, 2, 4,
        ]),
    ] {
        assert_eq!(
            select_ids(&ctx, table, filter).await,
            expected_ids,
            "WHERE {filter}"
        );
    }

    Ok(())
}
