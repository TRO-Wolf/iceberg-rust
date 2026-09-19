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
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::arrow::array::{Array, Int64Array, RecordBatch, TimestampMicrosecondArray};
use datafusion::arrow::datatypes::{
    DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef, TimeUnit,
};
use datafusion::catalog::Session;
use datafusion::datasource::{MemTable, TableProvider, TableType};
use datafusion::error::Result as DFResult;
use datafusion::execution::config::SessionConfig;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{col, lit};
use datafusion::scalar::ScalarValue;
use iceberg::expr::{Predicate, Reference};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{Datum, NestedField, PrimitiveType, Schema, Type};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use iceberg_datafusion::{IcebergCatalogProvider, IcebergTableProvider, IcebergTableScan};
use tempfile::TempDir;

const V1: i64 = 1_700_000_000_000_000;
const V2: i64 = 1_703_000_000_000_000;
const V3: i64 = 1_706_000_000_000_000;
const V4: i64 = -86_400_000_000;

#[derive(Debug)]
struct RecordingProvider {
    inner: IcebergTableProvider,
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl TableProvider for RecordingProvider {
    fn schema(&self) -> ArrowSchemaRef {
        self.inner.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        self.seen
            .lock()
            .expect("seen filters lock")
            .extend(filters.iter().map(|e| format!("scan filter: {e}")));
        self.inner.scan(state, projection, filters, limit).await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        self.seen
            .lock()
            .expect("seen filters lock")
            .extend(filters.iter().map(|e| format!("pushdown check: {e}")));
        self.inner.supports_filters_pushdown(filters)
    }

    async fn insert_into(
        &self,
        state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        self.inner.insert_into(state, input, insert_op).await
    }
}

struct Rig {
    ctx: SessionContext,
    seen: Arc<Mutex<Vec<String>>>,
    _warehouse: TempDir,
}

fn arrow_schema() -> ArrowSchemaRef {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new(
            "tsn",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ]))
}

fn row_batch(id: i64, micros: i64) -> RecordBatch {
    RecordBatch::try_new(arrow_schema(), vec![
        Arc::new(Int64Array::from(vec![id])),
        Arc::new(TimestampMicrosecondArray::from(vec![micros]).with_timezone("UTC")),
        Arc::new(TimestampMicrosecondArray::from(vec![micros])),
    ])
    .expect("row batch")
}

async fn insert_row(ctx: &SessionContext, ns: &str, id: i64, micros: i64) {
    let _ = ctx.deregister_table("src");
    ctx.register_table(
        "src",
        Arc::new(
            MemTable::try_new(arrow_schema(), vec![vec![row_batch(id, micros)]]).expect("memtable"),
        ),
    )
    .expect("register src");
    ctx.sql(&format!(
        "INSERT INTO catalog.{ns}.t SELECT id, ts, tsn FROM src"
    ))
    .await
    .expect("insert sql")
    .collect()
    .await
    .expect("insert");
}

async fn rig(ns: &str, target_partitions: usize) -> Rig {
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
        .await
        .expect("memory catalog");
    let namespace = NamespaceIdent::new(ns.to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "ts", Type::Primitive(PrimitiveType::Timestamptz)).into(),
            NestedField::optional(3, "tsn", Type::Primitive(PrimitiveType::Timestamp)).into(),
        ])
        .build()
        .expect("schema");
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("t".to_string())
                .location(format!("{warehouse_path}/t"))
                .schema(schema)
                .build(),
        )
        .await
        .expect("create table");
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_target_partitions(target_partitions),
    );
    let catalog_provider = Arc::new(
        IcebergCatalogProvider::try_new(catalog.clone())
            .await
            .expect("catalog provider"),
    );
    ctx.register_catalog("catalog", catalog_provider);
    for (id, micros) in [(1, V1), (2, V2), (3, V3), (4, V4)] {
        insert_row(&ctx, ns, id, micros).await;
    }
    let provider = IcebergTableProvider::try_new(catalog, namespace, "t")
        .await
        .expect("table provider");
    let seen = Arc::new(Mutex::new(Vec::new()));
    ctx.register_table(
        "t",
        Arc::new(RecordingProvider {
            inner: provider,
            seen: seen.clone(),
        }),
    )
    .expect("register t");
    ctx.register_table(
        "ref",
        Arc::new(
            MemTable::try_new(arrow_schema(), vec![vec![
                row_batch(1, V1),
                row_batch(2, V2),
                row_batch(3, V3),
                row_batch(4, V4),
            ]])
            .expect("ref memtable"),
        ),
    )
    .expect("register ref");
    Rig {
        ctx,
        seen,
        _warehouse: warehouse,
    }
}

async fn iceberg_plan(ctx: &SessionContext, filters: &[Expr]) -> Arc<dyn ExecutionPlan> {
    let provider = ctx.table_provider("t").await.expect("provider");
    let state = ctx.state();
    provider
        .scan(&state, None, filters, None)
        .await
        .expect("scan")
}

fn as_scan(plan: &Arc<dyn ExecutionPlan>) -> &IcebergTableScan {
    (plan.as_ref() as &dyn Any)
        .downcast_ref::<IcebergTableScan>()
        .expect("IcebergTableScan")
}

fn planned_files(plan: &Arc<dyn ExecutionPlan>) -> usize {
    as_scan(plan)
        .partition_work()
        .iter()
        .flat_map(|work| work.tasks())
        .count()
}

fn us(value: i64, tz: Option<&str>) -> Expr {
    lit(ScalarValue::TimestampMicrosecond(
        Some(value),
        tz.map(Into::into),
    ))
}

fn ns(value: i64, tz: Option<&str>) -> Expr {
    lit(ScalarValue::TimestampNanosecond(
        Some(value),
        tz.map(Into::into),
    ))
}

fn ts_cast(name: &str, unit: TimeUnit, tz: Option<&str>) -> Expr {
    Expr::Cast(datafusion::logical_expr::expr::Cast::new(
        Box::new(col(name)),
        DataType::Timestamp(unit, tz.map(Into::into)),
    ))
}

async fn sorted_ids(ctx: &SessionContext, table: &str, filter: &str) -> Vec<i64> {
    let batches = ctx
        .sql(&format!("SELECT id FROM {table} WHERE {filter}"))
        .await
        .expect("sql")
        .collect()
        .await
        .expect("collect");
    let mut ids: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id is Int64")
                .iter()
                .map(|value| value.expect("id is a required column"))
                .collect::<Vec<_>>()
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn unfiltered_scan_plans_every_file() {
    let rig = rig("ts_tz_all", 4).await;
    let plan = iceberg_plan(&rig.ctx, &[]).await;
    assert_eq!(planned_files(&plan), 4);
}

#[tokio::test]
async fn zoned_literals_push_a_predicate_and_prune_files() {
    let rig = rig("ts_tz_push", 4).await;
    let ts = Reference::new("ts");
    let cases: Vec<(Expr, Option<Predicate>, usize)> = vec![
        (
            col("ts").gt_eq(us(V2, Some("UTC"))),
            Some(
                ts.clone()
                    .greater_than_or_equal_to(Datum::timestamptz_micros(V2)),
            ),
            2,
        ),
        (
            col("ts").lt(us(V2, Some("UTC"))),
            Some(ts.clone().less_than(Datum::timestamptz_micros(V2))),
            2,
        ),
        (
            col("ts").eq(us(V2, Some("UTC"))),
            Some(ts.clone().equal_to(Datum::timestamptz_micros(V2))),
            1,
        ),
        (
            col("ts").gt_eq(us(V2, Some("+00:00"))),
            Some(
                ts.clone()
                    .greater_than_or_equal_to(Datum::timestamptz_micros(V2)),
            ),
            2,
        ),
        (
            col("ts").gt_eq(us(V2, Some("America/New_York"))),
            Some(
                ts.clone()
                    .greater_than_or_equal_to(Datum::timestamptz_micros(V2)),
            ),
            2,
        ),
        (
            col("ts").in_list(vec![us(V1, Some("UTC")), us(V3, Some("UTC"))], false),
            Some(
                ts.clone()
                    .is_in([Datum::timestamptz_micros(V1), Datum::timestamptz_micros(V3)]),
            ),
            3,
        ),
        (
            col("ts")
                .gt_eq(us(V2, Some("UTC")))
                .and(col("ts").lt_eq(us(V3, Some("UTC")))),
            Some(
                ts.clone()
                    .greater_than_or_equal_to(Datum::timestamptz_micros(V2))
                    .and(
                        ts.clone()
                            .less_than_or_equal_to(Datum::timestamptz_micros(V3)),
                    ),
            ),
            2,
        ),
        (
            col("ts").gt_eq(us(V4, Some("UTC"))),
            Some(
                ts.clone()
                    .greater_than_or_equal_to(Datum::timestamptz_micros(V4)),
            ),
            4,
        ),
        (
            col("ts").lt(us(V1, Some("UTC"))),
            Some(ts.clone().less_than(Datum::timestamptz_micros(V1))),
            1,
        ),
        (
            col("id").eq(lit(2_i64)),
            Some(Reference::new("id").equal_to(Datum::long(2))),
            1,
        ),
    ];
    for (expr, expected, files) in cases {
        let plan = iceberg_plan(&rig.ctx, std::slice::from_ref(&expr)).await;
        assert_eq!(
            as_scan(&plan).predicates(),
            expected.as_ref(),
            "pushed predicate for {expr}"
        );
        assert_eq!(planned_files(&plan), files, "planned files for {expr}");
    }
}

#[tokio::test]
async fn cross_zone_and_cross_unit_comparisons_stay_unpushed() {
    let rig = rig("ts_tz_nopush", 4).await;
    let cases = vec![
        col("ts").gt_eq(us(V2, None)),
        col("tsn").gt_eq(us(V2, Some("UTC"))),
        col("ts").gt_eq(ns(V2 * 1000, Some("UTC"))),
        ts_cast("ts", TimeUnit::Microsecond, None).gt_eq(us(V2, None)),
        ts_cast("tsn", TimeUnit::Microsecond, Some("UTC")).gt_eq(us(V2, Some("UTC"))),
        ts_cast("ts", TimeUnit::Nanosecond, Some("UTC")).gt_eq(ns(V2 * 1000, Some("UTC"))),
        ts_cast("tsn", TimeUnit::Nanosecond, None).gt_eq(ns(V2 * 1000, None)),
    ];
    for expr in cases {
        let plan = iceberg_plan(&rig.ctx, std::slice::from_ref(&expr)).await;
        assert_eq!(
            as_scan(&plan).predicates(),
            None,
            "{expr} must not reach the scan predicate"
        );
        assert_eq!(planned_files(&plan), 4, "{expr} must not prune files");
    }
    let expr = ts_cast("ts", TimeUnit::Microsecond, Some("+00:00")).gt_eq(us(V2, Some("+00:00")));
    let plan = iceberg_plan(&rig.ctx, std::slice::from_ref(&expr)).await;
    assert_eq!(
        as_scan(&plan).predicates(),
        Some(&Reference::new("ts").greater_than_or_equal_to(Datum::timestamptz_micros(V2))),
        "a zone-string-only cast keeps the instant and pushes"
    );
    assert_eq!(planned_files(&plan), 2);
}

#[tokio::test]
async fn zoneless_column_filters_still_push_and_prune() {
    let rig = rig("tsn_push", 4).await;
    let expr = col("tsn").gt_eq(us(V2, None));
    let plan = iceberg_plan(&rig.ctx, &[expr]).await;
    assert_eq!(
        as_scan(&plan).predicates(),
        Some(&Reference::new("tsn").greater_than_or_equal_to(Datum::timestamp_micros(V2)))
    );
    assert_eq!(planned_files(&plan), 2);
}

async fn plan_ids(ctx: &SessionContext, filter: Expr) -> Vec<i64> {
    let plan = iceberg_plan(ctx, &[filter]).await;
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
        .await
        .expect("collect plan");
    let mut ids: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id is Int64")
                .iter()
                .map(|value| value.expect("id is a required column"))
                .collect::<Vec<_>>()
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn boundary_row_survives_a_zoned_literal() {
    for target_partitions in [4, 1] {
        let rig = rig(
            &format!("ts_boundary_{target_partitions}"),
            target_partitions,
        )
        .await;
        for tz in ["UTC", "+00:00", "America/New_York"] {
            assert_eq!(
                plan_ids(&rig.ctx, col("ts").gt_eq(us(V2, Some(tz)))).await,
                vec![2, 3],
                "ts >= V2@{tz} (target_partitions={target_partitions})"
            );
            assert_eq!(
                plan_ids(&rig.ctx, col("ts").lt_eq(us(V2, Some(tz)))).await,
                vec![1, 2, 4],
                "ts <= V2@{tz} (target_partitions={target_partitions})"
            );
        }
    }
}

#[tokio::test]
async fn filtered_rows_match_the_in_memory_reference() {
    for target_partitions in [4, 1] {
        let rig = rig(&format!("ts_rows_{target_partitions}"), target_partitions).await;
        let cases = [
            ("ts >= CAST(1703000000 AS TIMESTAMP)", vec![2, 3]),
            ("ts < CAST(1703000000 AS TIMESTAMP)", vec![1, 4]),
            ("ts = CAST(1703000000 AS TIMESTAMP)", vec![2]),
            (
                "ts BETWEEN CAST(1703000000 AS TIMESTAMP) AND CAST(1706000000 AS TIMESTAMP)",
                vec![2, 3],
            ),
            (
                "ts IN (CAST(1700000000 AS TIMESTAMP), CAST(1706000000 AS TIMESTAMP))",
                vec![1, 3],
            ),
            ("ts >= CAST(-86400 AS TIMESTAMP)", vec![1, 2, 3, 4]),
            ("ts < CAST(1700000000 AS TIMESTAMP)", vec![4]),
            ("tsn >= CAST(1703000000 AS TIMESTAMP)", vec![2, 3]),
            ("id = 2", vec![2]),
        ];
        for (filter, expected) in cases {
            assert_eq!(
                sorted_ids(&rig.ctx, "t", filter).await,
                expected,
                "WHERE {filter} (target_partitions={target_partitions})"
            );
            assert_eq!(
                sorted_ids(&rig.ctx, "ref", filter).await,
                expected,
                "in-memory WHERE {filter}"
            );
        }
    }
}

#[tokio::test]
async fn record_observed_filter_exprs() {
    let rig = rig("ts_tz_observe", 4).await;
    let queries = [
        "SELECT id FROM t WHERE ts >= CAST(1703150000 AS TIMESTAMP)",
        "SELECT id FROM t WHERE ts < CAST(1703150000 AS TIMESTAMP)",
        "SELECT id FROM t WHERE ts = CAST(1703150000 AS TIMESTAMP)",
        "SELECT id FROM t WHERE ts BETWEEN CAST(1703150000 AS TIMESTAMP) AND CAST(1703160000 AS TIMESTAMP)",
        "SELECT id FROM t WHERE ts IN (CAST(1703150000 AS TIMESTAMP), CAST(1703160000 AS TIMESTAMP))",
        "SELECT id FROM t WHERE tsn >= CAST(1703150000 AS TIMESTAMP)",
        "SELECT id FROM t WHERE ts >= TIMESTAMP '2023-12-21 10:33:20'",
        "SELECT id FROM t WHERE ts >= CAST(1703150000 AS TIMESTAMP(6))",
        "SELECT id FROM t WHERE id = 5",
    ];
    for sql in queries {
        rig.seen.lock().expect("seen lock").clear();
        ctx_collect(&rig.ctx, sql).await;
        let seen = rig.seen.lock().expect("seen lock").join("\n");
        println!("{sql}\n{seen}\n");
    }
}

async fn ctx_collect(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .expect("sql")
        .collect()
        .await
        .expect("collect");
}
