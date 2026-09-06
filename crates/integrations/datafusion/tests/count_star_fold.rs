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

use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::common::stats::Precision;
use datafusion::execution::context::SessionContext;
use datafusion::prelude::{col, lit};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{FormatVersion, NestedField, PrimitiveType, Schema, Type};
use iceberg::{Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, TableCreation};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

fn temp_path() -> String {
    TempDir::new()
        .expect("temp dir")
        .path()
        .to_str()
        .expect("utf-8")
        .to_string()
}

async fn catalog() -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), temp_path())]),
        )
        .await
        .expect("memory catalog")
}

fn creation(location: &str, name: &str, mor: bool, version: FormatVersion) -> TableCreation {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    let properties = if mor {
        HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ])
    } else {
        HashMap::new()
    };
    TableCreation::builder()
        .location(location.to_string())
        .name(name.to_string())
        .schema(schema)
        .format_version(version)
        .properties(properties)
        .build()
}

async fn ctx_with_table(ns: &str, mor: bool, version: FormatVersion) -> (SessionContext, String) {
    let iceberg_catalog = catalog().await;
    let namespace = NamespaceIdent::new(ns.to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    iceberg_catalog
        .create_table(&namespace, creation(&temp_path(), "t", mor, version))
        .await
        .expect("create table");
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(Arc::new(iceberg_catalog))
            .await
            .expect("provider"),
    );
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", provider);
    (ctx, format!("catalog.{ns}.t"))
}

async fn sql_count(ctx: &SessionContext, sql: &str) -> i64 {
    let batches = ctx
        .sql(sql)
        .await
        .unwrap_or_else(|error| panic!("sql {sql}: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("collect {sql}: {error}"));
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count column")
        .value(0)
}

async fn physical_plan_text(ctx: &SessionContext, sql: &str) -> String {
    let batches = ctx
        .sql(sql)
        .await
        .expect("sql")
        .explain(false, false)
        .expect("explain")
        .collect()
        .await
        .expect("collect");
    batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("plan column")
        .value(1)
        .to_string()
}

async fn scan_plan(
    ctx: &SessionContext,
    table: &str,
    projection: Option<Vec<usize>>,
    filters: &[datafusion::logical_expr::Expr],
    limit: Option<usize>,
) -> Arc<dyn datafusion::physical_plan::ExecutionPlan> {
    let provider = ctx.table_provider(table).await.expect("table provider");
    let state = ctx.state();
    provider
        .scan(&state, projection.as_ref(), filters, limit)
        .await
        .expect("scan")
}

#[tokio::test]
async fn count_star_folds_on_plain_table() {
    let (ctx, table) = ctx_with_table("fold_plain", false, FormatVersion::V2).await;
    ctx.sql(&format!(
        "INSERT INTO {table} VALUES (1, 'a'), (2, 'b'), (3, 'c')"
    ))
    .await
    .expect("insert")
    .collect()
    .await
    .expect("collect");
    let plan = scan_plan(&ctx, &table, Some(vec![]), &[], None).await;
    let stats = plan.partition_statistics(None).expect("statistics");
    assert!(matches!(stats.num_rows, Precision::Exact(3)));
    assert_eq!(
        sql_count(&ctx, &format!("SELECT count(*) FROM {table}")).await,
        3
    );
    let physical = physical_plan_text(&ctx, &format!("SELECT count(*) FROM {table}")).await;
    assert!(
        !physical.contains("IcebergTableScan"),
        "folded plan must not scan: {physical}"
    );
}

#[tokio::test]
async fn count_star_does_not_fold_with_deletes() {
    let (ctx, table) = ctx_with_table("fold_mor", true, FormatVersion::V3).await;
    ctx.sql(&format!(
        "INSERT INTO {table} VALUES (1, 'a'), (2, 'b'), (3, 'c')"
    ))
    .await
    .expect("insert")
    .collect()
    .await
    .expect("collect");
    ctx.sql(&format!("DELETE FROM {table} WHERE foo1 = 2"))
        .await
        .expect("delete")
        .collect()
        .await
        .expect("collect");
    let plan = scan_plan(&ctx, &table, Some(vec![]), &[], None).await;
    let stats = plan.partition_statistics(None).expect("statistics");
    assert!(matches!(stats.num_rows, Precision::Absent));
    assert_eq!(
        sql_count(&ctx, &format!("SELECT count(*) FROM {table}")).await,
        2
    );
    let physical = physical_plan_text(&ctx, &format!("SELECT count(*) FROM {table}")).await;
    assert!(
        physical.contains("IcebergTableScan"),
        "unfolded plan must scan: {physical}"
    );
}

#[tokio::test]
async fn count_star_does_not_fold_with_residual() {
    let (ctx, table) = ctx_with_table("fold_where", false, FormatVersion::V2).await;
    ctx.sql(&format!(
        "INSERT INTO {table} VALUES (1, 'a'), (2, 'b'), (3, 'c')"
    ))
    .await
    .expect("insert")
    .collect()
    .await
    .expect("collect");
    let filter = col("foo1").gt(lit(1));
    let plan = scan_plan(&ctx, &table, Some(vec![]), &[filter], None).await;
    let stats = plan.partition_statistics(None).expect("statistics");
    assert!(matches!(stats.num_rows, Precision::Absent));
    assert_eq!(
        sql_count(
            &ctx,
            &format!("SELECT count(*) FROM {table} WHERE foo1 > 1")
        )
        .await,
        2
    );
}

#[tokio::test]
async fn scan_statistics_unknown_with_limit() {
    let (ctx, table) = ctx_with_table("fold_limit", false, FormatVersion::V2).await;
    ctx.sql(&format!(
        "INSERT INTO {table} VALUES (1, 'a'), (2, 'b'), (3, 'c')"
    ))
    .await
    .expect("insert")
    .collect()
    .await
    .expect("collect");
    let plan = scan_plan(&ctx, &table, Some(vec![0]), &[], Some(1)).await;
    let stats = plan.partition_statistics(None).expect("statistics");
    assert!(matches!(stats.num_rows, Precision::Absent));
}

#[tokio::test]
async fn count_star_on_empty_table_is_zero() {
    let (ctx, table) = ctx_with_table("fold_empty", false, FormatVersion::V2).await;
    let plan = scan_plan(&ctx, &table, Some(vec![]), &[], None).await;
    let stats = plan.partition_statistics(None).expect("statistics");
    assert!(matches!(stats.num_rows, Precision::Exact(0)));
    assert_eq!(
        sql_count(&ctx, &format!("SELECT count(*) FROM {table}")).await,
        0
    );
}

#[tokio::test]
async fn count_star_stays_folded_after_cow_delete() {
    let (ctx, table) = ctx_with_table("fold_cow", false, FormatVersion::V2).await;
    ctx.sql(&format!(
        "INSERT INTO {table} VALUES (1, 'a'), (2, 'b'), (3, 'c')"
    ))
    .await
    .expect("insert")
    .collect()
    .await
    .expect("collect");
    ctx.sql(&format!("DELETE FROM {table} WHERE foo1 = 2"))
        .await
        .expect("delete")
        .collect()
        .await
        .expect("collect");
    let plan = scan_plan(&ctx, &table, Some(vec![]), &[], None).await;
    let stats = plan.partition_statistics(None).expect("statistics");
    assert!(matches!(stats.num_rows, Precision::Exact(2)));
    assert_eq!(
        sql_count(&ctx, &format!("SELECT count(*) FROM {table}")).await,
        2
    );
}

#[tokio::test]
async fn count_star_partition_statistics_are_whole_table_only() {
    let (ctx, table) = ctx_with_table("fold_parts", false, FormatVersion::V2).await;
    ctx.sql(&format!(
        "INSERT INTO {table} VALUES (1, 'a'), (2, 'b'), (3, 'c')"
    ))
    .await
    .expect("insert")
    .collect()
    .await
    .expect("collect");
    let plan = scan_plan(&ctx, &table, Some(vec![]), &[], None).await;
    let stats = plan.partition_statistics(Some(0)).expect("statistics");
    assert!(matches!(stats.num_rows, Precision::Absent));
}
