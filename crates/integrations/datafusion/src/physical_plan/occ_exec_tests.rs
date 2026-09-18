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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, UInt64Array};
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::datasource::TableProvider;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{Expr, SessionContext, col, lit};
use futures::TryStreamExt;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal,
    NestedField, PrimitiveType, Schema as IcebergSchema, Struct, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{
    Catalog, CatalogBuilder, Error, ErrorKind, NamespaceIdent, TableCreation, TableIdent,
};
use tempfile::TempDir;

use crate::IcebergCatalogProvider;
use crate::table::IcebergTableProvider;

struct OccFixture {
    catalog: Arc<MemoryCatalog>,
    provider: IcebergTableProvider,
    ident: TableIdent,
    ctx: SessionContext,
    _warehouse: TempDir,
}

impl OccFixture {
    async fn load_table(&self) -> Table {
        self.catalog
            .load_table(&self.ident)
            .await
            .expect("load table")
    }
}

async fn occ_fixture(merge_on_read: bool) -> OccFixture {
    let warehouse = TempDir::new().expect("warehouse");
    let catalog = Arc::new(
        MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    warehouse.path().to_str().expect("utf8").to_string(),
                )]),
            )
            .await
            .expect("catalog"),
    );
    let namespace = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "k", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "v", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "k", Transform::Identity)
        .expect("identity(k)")
        .build();
    let properties = if merge_on_read {
        HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ])
    } else {
        HashMap::new()
    };
    let ident = TableIdent::new(namespace.clone(), "t".to_string());
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("t".to_string())
                .location(format!("{}/t", warehouse.path().to_str().expect("utf8")))
                .schema(schema)
                .partition_spec(spec)
                .format_version(FormatVersion::V2)
                .properties(properties)
                .build(),
        )
        .await
        .expect("table");

    let catalog_provider = IcebergCatalogProvider::try_new(catalog.clone())
        .await
        .expect("catalog provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(catalog_provider));
    ctx.sql(
        "INSERT INTO catalog.ns.t VALUES \
         (1,'a','v1'),(2,'a','v2'),(3,'b','v3'),(4,'b','v4')",
    )
    .await
    .expect("plan seed insert")
    .collect()
    .await
    .expect("seed insert");

    let provider = IcebergTableProvider::try_new(catalog.clone(), namespace, "t")
        .await
        .expect("table provider");
    OccFixture {
        catalog,
        provider,
        ident,
        ctx,
        _warehouse: warehouse,
    }
}

async fn float_fixture(merge_on_read: bool, f_seed: &str) -> OccFixture {
    let warehouse = TempDir::new().expect("warehouse");
    let catalog = Arc::new(
        MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    warehouse.path().to_str().expect("utf8").to_string(),
                )]),
            )
            .await
            .expect("catalog"),
    );
    let namespace = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "f", Type::Primitive(PrimitiveType::Float)).into(),
        ])
        .build()
        .expect("schema");
    let spec = UnboundPartitionSpec::builder().with_spec_id(0).build();
    let properties = if merge_on_read {
        HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ])
    } else {
        HashMap::new()
    };
    let ident = TableIdent::new(namespace.clone(), "t".to_string());
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("t".to_string())
                .location(format!("{}/t", warehouse.path().to_str().expect("utf8")))
                .schema(schema)
                .partition_spec(spec)
                .format_version(FormatVersion::V2)
                .properties(properties)
                .build(),
        )
        .await
        .expect("table");

    let catalog_provider = IcebergCatalogProvider::try_new(catalog.clone())
        .await
        .expect("catalog provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(catalog_provider));
    ctx.sql(&format!("INSERT INTO catalog.ns.t VALUES {f_seed}"))
        .await
        .expect("plan seed insert")
        .collect()
        .await
        .expect("seed insert");

    let provider = IcebergTableProvider::try_new(catalog.clone(), namespace, "t")
        .await
        .expect("table provider");
    OccFixture {
        catalog,
        provider,
        ident,
        ctx,
        _warehouse: warehouse,
    }
}

fn where_k_a_id_lt_8() -> Vec<Expr> {
    vec![col("k").eq(lit("a")), col("id").lt(lit(8))]
}

fn update_v() -> Vec<(String, Expr)> {
    vec![("v".to_string(), lit("u"))]
}

fn concurrent_data_file(name: &str, partition: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("test/{name}.parquet"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::string(partition))]))
        .build()
        .expect("concurrent data file")
}

fn concurrent_delete_file(name: &str, partition: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(format!("test/{name}.parquet"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::string(partition))]))
        .build()
        .expect("concurrent delete file")
}

async fn concurrent_append(fixture: &OccFixture, base: &Table, file: DataFile) {
    let tx = Transaction::new(base);
    tx.fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply concurrent append")
        .commit(fixture.catalog.as_ref())
        .await
        .expect("concurrent append commits");
}

async fn concurrent_row_delta(
    fixture: &OccFixture,
    base: &Table,
    data: DataFile,
    delete: DataFile,
) {
    let tx = Transaction::new(base);
    tx.row_delta()
        .add_data_files(vec![data])
        .add_deletes(vec![delete])
        .apply(tx)
        .expect("apply concurrent row delta")
        .commit(fixture.catalog.as_ref())
        .await
        .expect("concurrent row delta commits");
}

async fn concurrent_deletes(fixture: &OccFixture, base: &Table, delete: DataFile) {
    let tx = Transaction::new(base);
    tx.row_delta()
        .add_deletes(vec![delete])
        .apply(tx)
        .expect("apply concurrent deletes")
        .commit(fixture.catalog.as_ref())
        .await
        .expect("concurrent deletes commit");
}

async fn run_dml(exec: Arc<dyn ExecutionPlan>) -> DFResult<u64> {
    let mut stream = exec.execute(0, Arc::new(TaskContext::default()))?;
    let batch = stream
        .try_next()
        .await?
        .ok_or_else(|| DataFusionError::Internal("DML produced no count batch".to_string()))?;
    let count = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("count column");
    Ok(count.value(0))
}

fn assert_conflict(err: &DataFusionError, needle: &str) {
    let text = err.to_string();
    assert!(
        text.contains("conflicting"),
        "expected a serializable conflict, got: {text}"
    );
    assert!(
        text.contains(needle),
        "the conflict must name {needle}, got: {text}"
    );
    let DataFusionError::External(source) = err else {
        panic!("expected an iceberg error, got: {err:?}");
    };
    let inner = source
        .downcast_ref::<Error>()
        .expect("iceberg error source");
    assert_eq!(inner.kind(), ErrorKind::DataInvalid);
    assert!(!inner.retryable());
}

async fn live_paths(table: &Table) -> HashSet<String> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table should have a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list should load");
    let mut live = HashSet::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest should load");
        for entry in manifest.entries() {
            if entry.is_alive() {
                live.insert(entry.file_path().to_string());
            }
        }
    }
    live
}

#[tokio::test]
async fn mor_delete_disjoint_partition_commit_commits() {
    let fixture = occ_fixture(true).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .delete_from(&state, where_k_a_id_lt_8())
        .await
        .expect("plan DELETE WHERE k = 'a' AND id < 8");
    let base = fixture.load_table().await;
    concurrent_row_delta(
        &fixture,
        &base,
        concurrent_data_file("b-new", "b"),
        concurrent_delete_file("b-del", "b"),
    )
    .await;
    let deleted = run_dml(exec)
        .await
        .expect("a concurrent commit in a disjoint partition must not conflict the k = 'a' DELETE");
    assert_eq!(deleted, 2);
    let live = live_paths(&fixture.load_table().await).await;
    assert!(live.iter().any(|path| path.contains("b-new.parquet")));
}

#[tokio::test]
async fn mor_delete_matching_partition_commit_conflicts() {
    let fixture = occ_fixture(true).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .delete_from(&state, where_k_a_id_lt_8())
        .await
        .expect("plan DELETE WHERE k = 'a' AND id < 8");
    let base = fixture.load_table().await;
    concurrent_append(&fixture, &base, concurrent_data_file("a-new", "a")).await;
    let err = run_dml(exec).await.expect_err(
        "a concurrent commit in the filtered partition must conflict the k = 'a' DELETE",
    );
    assert_conflict(&err, "a-new.parquet");
}

#[tokio::test]
async fn cow_delete_disjoint_partition_commit_commits() {
    let fixture = occ_fixture(false).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .delete_from(&state, where_k_a_id_lt_8())
        .await
        .expect("plan DELETE WHERE k = 'a' AND id < 8");
    let base = fixture.load_table().await;
    concurrent_append(&fixture, &base, concurrent_data_file("b-new", "b")).await;
    let deleted = run_dml(exec)
        .await
        .expect("a concurrent commit in a disjoint partition must not conflict the k = 'a' DELETE");
    assert_eq!(deleted, 2);
    let live = live_paths(&fixture.load_table().await).await;
    assert!(live.iter().any(|path| path.contains("b-new.parquet")));
}

#[tokio::test]
async fn cow_delete_matching_partition_commit_conflicts() {
    let fixture = occ_fixture(false).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .delete_from(&state, where_k_a_id_lt_8())
        .await
        .expect("plan DELETE WHERE k = 'a' AND id < 8");
    let base = fixture.load_table().await;
    concurrent_append(&fixture, &base, concurrent_data_file("a-new", "a")).await;
    let err = run_dml(exec).await.expect_err(
        "a concurrent commit in the filtered partition must conflict the k = 'a' DELETE",
    );
    assert_conflict(&err, "a-new.parquet");
}

#[tokio::test]
async fn mor_update_disjoint_partition_commit_commits() {
    let fixture = occ_fixture(true).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .update(&state, update_v(), where_k_a_id_lt_8())
        .await
        .expect("plan UPDATE WHERE k = 'a' AND id < 8");
    let base = fixture.load_table().await;
    concurrent_row_delta(
        &fixture,
        &base,
        concurrent_data_file("b-new", "b"),
        concurrent_delete_file("b-del", "b"),
    )
    .await;
    let updated = run_dml(exec)
        .await
        .expect("a concurrent commit in a disjoint partition must not conflict the k = 'a' UPDATE");
    assert_eq!(updated, 2);
    let live = live_paths(&fixture.load_table().await).await;
    assert!(live.iter().any(|path| path.contains("b-new.parquet")));
}

#[tokio::test]
async fn mor_update_matching_partition_commit_conflicts() {
    let fixture = occ_fixture(true).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .update(&state, update_v(), where_k_a_id_lt_8())
        .await
        .expect("plan UPDATE WHERE k = 'a' AND id < 8");
    let base = fixture.load_table().await;
    concurrent_row_delta(
        &fixture,
        &base,
        concurrent_data_file("a-new", "a"),
        concurrent_delete_file("a-del", "a"),
    )
    .await;
    let err = run_dml(exec).await.expect_err(
        "a concurrent commit in the filtered partition must conflict the k = 'a' UPDATE",
    );
    assert_conflict(&err, "a-new.parquet");
}

#[tokio::test]
async fn mor_update_matching_partition_delete_file_conflicts() {
    let fixture = occ_fixture(true).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .update(&state, update_v(), where_k_a_id_lt_8())
        .await
        .expect("plan UPDATE WHERE k = 'a' AND id < 8");
    let base = fixture.load_table().await;
    concurrent_deletes(&fixture, &base, concurrent_delete_file("a-del", "a")).await;
    let err = run_dml(exec).await.expect_err(
        "a concurrent delete file in the filtered partition must conflict the k = 'a' UPDATE",
    );
    assert_conflict(&err, "a-del.parquet");
}

#[tokio::test]
async fn cow_update_disjoint_partition_commit_commits() {
    let fixture = occ_fixture(false).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .update(&state, update_v(), where_k_a_id_lt_8())
        .await
        .expect("plan UPDATE WHERE k = 'a' AND id < 8");
    let base = fixture.load_table().await;
    concurrent_append(&fixture, &base, concurrent_data_file("b-new", "b")).await;
    let updated = run_dml(exec)
        .await
        .expect("a concurrent commit in a disjoint partition must not conflict the k = 'a' UPDATE");
    assert_eq!(updated, 2);
    let live = live_paths(&fixture.load_table().await).await;
    assert!(live.iter().any(|path| path.contains("b-new.parquet")));
}

#[tokio::test]
async fn cow_update_matching_partition_commit_conflicts() {
    let fixture = occ_fixture(false).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .update(&state, update_v(), where_k_a_id_lt_8())
        .await
        .expect("plan UPDATE WHERE k = 'a' AND id < 8");
    let base = fixture.load_table().await;
    concurrent_append(&fixture, &base, concurrent_data_file("a-new", "a")).await;
    let err = run_dml(exec).await.expect_err(
        "a concurrent commit in the filtered partition must conflict the k = 'a' UPDATE",
    );
    assert_conflict(&err, "a-new.parquet");
}

#[tokio::test]
async fn mor_delete_no_predicate_keeps_always_true() {
    let fixture = occ_fixture(true).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .delete_from(&state, vec![])
        .await
        .expect("plan DELETE FROM t");
    let base = fixture.load_table().await;
    concurrent_append(&fixture, &base, concurrent_data_file("b-new", "b")).await;
    let err = run_dml(exec).await.expect_err(
        "a predicate-less DELETE keeps AlwaysTrue and must conflict any concurrent commit",
    );
    let text = err.to_string();
    assert!(
        text.contains("matching TRUE"),
        "expected AlwaysTrue: {text}"
    );
    assert_conflict(&err, "b-new.parquet");
}

#[tokio::test]
async fn cow_delete_no_predicate_keeps_always_true() {
    let fixture = occ_fixture(false).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .delete_from(&state, vec![])
        .await
        .expect("plan DELETE FROM t");
    let base = fixture.load_table().await;
    concurrent_append(&fixture, &base, concurrent_data_file("b-new", "b")).await;
    let err = run_dml(exec).await.expect_err(
        "a predicate-less DELETE keeps AlwaysTrue and must conflict any concurrent commit",
    );
    let text = err.to_string();
    assert!(
        text.contains("matching TRUE"),
        "expected AlwaysTrue: {text}"
    );
    assert_conflict(&err, "b-new.parquet");
}

#[tokio::test]
async fn mor_update_no_predicate_keeps_always_true() {
    let fixture = occ_fixture(true).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .update(&state, update_v(), vec![])
        .await
        .expect("plan UPDATE FROM t");
    let base = fixture.load_table().await;
    concurrent_append(&fixture, &base, concurrent_data_file("b-new", "b")).await;
    let err = run_dml(exec).await.expect_err(
        "a predicate-less UPDATE keeps AlwaysTrue and must conflict any concurrent commit",
    );
    let text = err.to_string();
    assert!(
        text.contains("matching TRUE"),
        "expected AlwaysTrue: {text}"
    );
    assert_conflict(&err, "b-new.parquet");
}

#[tokio::test]
async fn cow_update_no_predicate_keeps_always_true() {
    let fixture = occ_fixture(false).await;
    let state = fixture.ctx.state();
    let exec = fixture
        .provider
        .update(&state, update_v(), vec![])
        .await
        .expect("plan UPDATE FROM t");
    let base = fixture.load_table().await;
    concurrent_append(&fixture, &base, concurrent_data_file("b-new", "b")).await;
    let err = run_dml(exec).await.expect_err(
        "a predicate-less UPDATE keeps AlwaysTrue and must conflict any concurrent commit",
    );
    let text = err.to_string();
    assert!(
        text.contains("matching TRUE"),
        "expected AlwaysTrue: {text}"
    );
    assert_conflict(&err, "b-new.parquet");
}

async fn count_rows(ctx: &SessionContext, sql: &str) -> i64 {
    let batches = ctx
        .sql(sql)
        .await
        .expect("plan count query")
        .collect()
        .await
        .expect("run count query");
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("count column")
                .values()
                .iter()
                .copied()
        })
        .sum()
}

async fn dml_count(ctx: &SessionContext, sql: &str) -> u64 {
    let batches = ctx
        .sql(sql)
        .await
        .expect("plan DML")
        .collect()
        .await
        .expect("DML commits");
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("count column")
                .values()
                .iter()
                .copied()
        })
        .sum()
}

#[tokio::test]
async fn delete_where_float_lt_inexact_double_deletes_the_row() {
    for merge_on_read in [true, false] {
        let fixture = float_fixture(merge_on_read, "(1, 1.0)").await;
        let deleted = dml_count(
            &fixture.ctx,
            "DELETE FROM catalog.ns.t WHERE f < 1.00000001",
        )
        .await;
        assert_eq!(
            deleted, 1,
            "merge_on_read={merge_on_read}: f = 1.0 satisfies f < 1.00000001"
        );
        let remaining = count_rows(&fixture.ctx, "SELECT COUNT(*) FROM catalog.ns.t").await;
        assert_eq!(
            remaining, 0,
            "merge_on_read={merge_on_read}: the row must be deleted"
        );
    }
}

#[tokio::test]
async fn update_where_float_lt_inexact_double_updates_the_row() {
    for merge_on_read in [true, false] {
        let fixture = float_fixture(merge_on_read, "(1, 1.0)").await;
        let updated = dml_count(
            &fixture.ctx,
            "UPDATE catalog.ns.t SET id = 9 WHERE f < 1.00000001",
        )
        .await;
        assert_eq!(
            updated, 1,
            "merge_on_read={merge_on_read}: f = 1.0 satisfies f < 1.00000001"
        );
        let matched = count_rows(
            &fixture.ctx,
            "SELECT COUNT(*) FROM catalog.ns.t WHERE id = 9",
        )
        .await;
        assert_eq!(
            matched, 1,
            "merge_on_read={merge_on_read}: the row must carry id = 9"
        );
    }
}

#[tokio::test]
async fn delete_where_cast_to_int_eq_deletes_the_row() {
    for merge_on_read in [true, false] {
        let fixture = float_fixture(merge_on_read, "(1, 2.5)").await;
        let deleted = dml_count(
            &fixture.ctx,
            "DELETE FROM catalog.ns.t WHERE CAST(f AS INT) = 2",
        )
        .await;
        assert_eq!(
            deleted, 1,
            "merge_on_read={merge_on_read}: CAST(2.5 AS INT) = 2 is true"
        );
        let remaining = count_rows(&fixture.ctx, "SELECT COUNT(*) FROM catalog.ns.t").await;
        assert_eq!(
            remaining, 0,
            "merge_on_read={merge_on_read}: the row must be deleted"
        );
    }
}

#[tokio::test]
async fn delete_where_float_gt_beyond_f32_max_deletes_the_row() {
    for merge_on_read in [true, false] {
        let fixture = float_fixture(merge_on_read, "(1, CAST('inf' AS REAL))").await;
        let deleted = dml_count(&fixture.ctx, "DELETE FROM catalog.ns.t WHERE f > 3.5e38").await;
        assert_eq!(
            deleted, 1,
            "merge_on_read={merge_on_read}: f = +Inf satisfies f > 3.5e38"
        );
        let remaining = count_rows(&fixture.ctx, "SELECT COUNT(*) FROM catalog.ns.t").await;
        assert_eq!(
            remaining, 0,
            "merge_on_read={merge_on_read}: the row must be deleted"
        );
    }
}

#[tokio::test]
async fn delete_where_float_le_negative_zero_deletes_only_negative_zero() {
    for merge_on_read in [true, false] {
        let fixture = float_fixture(merge_on_read, "(1, -0.0), (2, 0.0)").await;
        let deleted = dml_count(&fixture.ctx, "DELETE FROM catalog.ns.t WHERE f <= -0.0").await;
        assert_eq!(
            deleted, 1,
            "merge_on_read={merge_on_read}: DataFusion total order matches only the -0.0 row"
        );
        let remaining = count_rows(
            &fixture.ctx,
            "SELECT COUNT(*) FROM catalog.ns.t WHERE f = 0.0",
        )
        .await;
        assert_eq!(
            remaining, 1,
            "merge_on_read={merge_on_read}: the +0.0 row must survive"
        );
    }
}

#[tokio::test]
async fn delete_where_float_ne_nan_deletes_only_non_nan_rows() {
    for merge_on_read in [true, false] {
        let fixture = float_fixture(merge_on_read, "(1, 0.0), (2, CAST('NaN' AS REAL))").await;
        let deleted = dml_count(
            &fixture.ctx,
            "DELETE FROM catalog.ns.t WHERE f != CAST('NaN' AS DOUBLE)",
        )
        .await;
        assert_eq!(
            deleted, 1,
            "merge_on_read={merge_on_read}: only the 0.0 row satisfies f != NaN"
        );
        let remaining = count_rows(
            &fixture.ctx,
            "SELECT COUNT(*) FROM catalog.ns.t WHERE f = CAST('NaN' AS DOUBLE)",
        )
        .await;
        assert_eq!(
            remaining, 1,
            "merge_on_read={merge_on_read}: the NaN row must survive"
        );
    }
}
