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
use std::sync::Arc;

use datafusion::arrow::array::Int32Array;
use datafusion::common::stats::Precision;
use datafusion::execution::TaskContext;
use datafusion::execution::config::SessionConfig;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use iceberg_datafusion::{IcebergCatalogProvider, IcebergTableScan};
use tempfile::TempDir;

fn temp_path() -> String {
    TempDir::new()
        .expect("temp dir")
        .path()
        .to_str()
        .expect("utf-8")
        .to_string()
}

async fn ctx_with_files(
    ns: &str,
    files: usize,
    target_partitions: usize,
) -> (SessionContext, String) {
    let warehouse = temp_path();
    let iceberg_catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse)]),
        )
        .await
        .expect("memory catalog");
    let namespace = NamespaceIdent::new(ns.to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    iceberg_catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .location(temp_path())
                .name("t".to_string())
                .schema(schema)
                .build(),
        )
        .await
        .expect("create table");
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(Arc::new(iceberg_catalog))
            .await
            .expect("provider"),
    );
    let config = SessionConfig::new().with_target_partitions(target_partitions);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_catalog("catalog", provider);
    let table = format!("catalog.{ns}.t");
    for file in 0..files {
        let base = (file * 3) as i32;
        ctx.sql(&format!(
            "INSERT INTO {table} VALUES ({}, 'a'), ({}, 'b'), ({}, 'c')",
            base + 1,
            base + 2,
            base + 3
        ))
        .await
        .expect("insert")
        .collect()
        .await
        .expect("collect");
    }
    (ctx, table)
}

async fn scan_partition_count(
    ctx: &SessionContext,
    table: &str,
    projection: Option<Vec<usize>>,
) -> usize {
    let provider = ctx.table_provider(table).await.expect("table provider");
    let state = ctx.state();
    let plan = provider
        .scan(&state, projection.as_ref(), &[], None)
        .await
        .expect("scan");
    plan.properties().output_partitioning().partition_count()
}

async fn scan_with_limit(
    ctx: &SessionContext,
    table: &str,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
) -> Arc<dyn ExecutionPlan> {
    let provider = ctx.table_provider(table).await.expect("table provider");
    let state = ctx.state();
    provider
        .scan(&state, projection.as_ref(), &[], limit)
        .await
        .expect("scan")
}

async fn sorted_ids(ctx: &SessionContext, table: &str) -> Vec<i32> {
    let batches = ctx
        .sql(&format!("SELECT foo1 FROM {table} ORDER BY foo1"))
        .await
        .expect("sql")
        .collect()
        .await
        .expect("collect");
    let mut ids: Vec<i32> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id column")
                .values()
                .to_vec()
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn small_table_scans_in_parallel() {
    let (ctx, table) = ctx_with_files("par_eight", 8, 8).await;
    assert_eq!(
        scan_partition_count(&ctx, &table, Some(vec![0, 1])).await,
        8
    );
    let expected: Vec<i32> = (1..=24).collect();
    assert_eq!(sorted_ids(&ctx, &table).await, expected);
}

#[tokio::test]
async fn small_table_scan_is_single_partition_at_target_one() {
    let (ctx, table) = ctx_with_files("par_one", 8, 1).await;
    assert_eq!(
        scan_partition_count(&ctx, &table, Some(vec![0, 1])).await,
        1
    );
    let expected: Vec<i32> = (1..=24).collect();
    assert_eq!(sorted_ids(&ctx, &table).await, expected);
}

#[tokio::test]
async fn limit_cleared_at_n_gt_1_scan_emits_all_rows_with_exact_stats() {
    let (ctx, table) = ctx_with_files("par_limit", 8, 8).await;
    let plan = scan_with_limit(&ctx, &table, Some(vec![0, 1]), Some(2)).await;
    assert_eq!(plan.properties().output_partitioning().partition_count(), 8);
    let scan = (&*plan as &dyn Any)
        .downcast_ref::<IcebergTableScan>()
        .expect("IcebergTableScan");
    assert_eq!(scan.limit(), None);
    let stats = plan.partition_statistics(None).expect("statistics");
    assert!(matches!(stats.num_rows, Precision::Exact(24)));
    let task_ctx = Arc::new(TaskContext::default());
    let mut total = 0;
    for partition in 0..8 {
        let batches: Vec<_> = plan
            .execute(partition, task_ctx.clone())
            .expect("execute")
            .try_collect()
            .await
            .expect("collect");
        let rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(rows, 3);
        total += rows;
    }
    assert_eq!(total, 24);
    let batches = ctx
        .sql(&format!("SELECT foo1 FROM {table} ORDER BY foo1 LIMIT 2"))
        .await
        .expect("sql")
        .collect()
        .await
        .expect("collect");
    let ids: Vec<i32> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id column")
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(ids, vec![1, 2]);
}

#[tokio::test]
async fn empty_projection_scan_stays_single_partition() {
    let (ctx, table) = ctx_with_files("par_empty", 8, 8).await;
    assert_eq!(scan_partition_count(&ctx, &table, Some(vec![])).await, 1);
    let batches = ctx
        .sql(&format!("SELECT count(*) FROM {table}"))
        .await
        .expect("sql")
        .collect()
        .await
        .expect("collect");
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1
    );
}
