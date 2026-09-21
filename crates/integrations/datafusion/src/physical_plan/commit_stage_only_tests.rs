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

use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, NamespaceIdent, TableIdent};

use super::IcebergCommitExec;
use super::tests::{
    BoxResult, TestResult, append_files_direct, assert_count, data_file_json, live_paths_in,
    make_data_file, run_commit_exec, setup_table,
};
use crate::table::IcebergTableProvider;

async fn commit_stage_only_of(
    provider: &IcebergTableProvider,
    state: &dyn datafusion::catalog::Session,
) -> BoxResult<bool> {
    use datafusion::datasource::TableProvider;
    use datafusion::physical_plan::empty::EmptyExec;
    let input = Arc::new(EmptyExec::new(provider.schema())) as Arc<dyn ExecutionPlan>;
    let plan = provider.insert_into(state, input, InsertOp::Append).await?;
    Ok(plan
        .downcast_ref::<IcebergCommitExec>()
        .expect("insert_into plans IcebergCommitExec")
        .stage_only)
}

#[tokio::test]
async fn test_provider_stage_only_defaults_false_and_threads() -> TestResult {
    let (catalog, _) = setup_table(HashMap::new()).await?;
    let ctx = SessionContext::new();
    let state = ctx.state();
    let default = IcebergTableProvider::try_new(
        Arc::clone(&catalog),
        NamespaceIdent::new("ns".to_string()),
        "t".to_string(),
    )
    .await?;
    assert!(!commit_stage_only_of(&default, &state).await?);
    let staged = IcebergTableProvider::try_new(
        Arc::clone(&catalog),
        NamespaceIdent::new("ns".to_string()),
        "t".to_string(),
    )
    .await?
    .with_stage_only(true);
    assert!(commit_stage_only_of(&staged, &state).await?);
    Ok(())
}

#[tokio::test]
async fn test_append_stage_only_adds_snapshot_without_moving_current() -> TestResult {
    let (catalog, table) = setup_table(HashMap::new()).await?;
    let table = append_files_direct(&catalog, &table, vec![make_data_file(
        &table,
        "base.parquet",
        7,
    )?])
    .await?;
    let head = table.metadata().current_snapshot_id().expect("seeded head");
    let new_json = data_file_json(&table, make_data_file(&table, "new.parquet", 42)?)?;
    let batches = run_commit_exec(
        &table,
        &catalog,
        vec![new_json],
        InsertOp::Append,
        true,
        None,
        HashMap::new(),
    )
    .await?;
    assert_count(&batches, 42);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "t"])?)
        .await?;
    assert_eq!(reloaded.metadata().snapshots().len(), 2);
    assert_eq!(reloaded.metadata().current_snapshot_id(), Some(head));
    let staged = reloaded
        .metadata()
        .snapshots()
        .find(|s| s.snapshot_id() != head)
        .expect("staged snapshot");
    assert_eq!(live_paths_in(&reloaded, staged).await?, vec![
        "base.parquet",
        "new.parquet"
    ]);
    Ok(())
}

#[tokio::test]
async fn test_overwrite_stage_only_adds_snapshot_without_moving_current() -> TestResult {
    let (catalog, table) = setup_table(HashMap::new()).await?;
    let table = append_files_direct(&catalog, &table, vec![make_data_file(
        &table,
        "base.parquet",
        7,
    )?])
    .await?;
    let head = table.metadata().current_snapshot_id().expect("seeded head");
    let new_json = data_file_json(&table, make_data_file(&table, "new.parquet", 42)?)?;
    let batches = run_commit_exec(
        &table,
        &catalog,
        vec![new_json],
        InsertOp::Overwrite,
        true,
        None,
        HashMap::new(),
    )
    .await?;
    assert_count(&batches, 42);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "t"])?)
        .await?;
    assert_eq!(reloaded.metadata().snapshots().len(), 2);
    assert_eq!(reloaded.metadata().current_snapshot_id(), Some(head));
    let staged = reloaded
        .metadata()
        .snapshots()
        .find(|s| s.snapshot_id() != head)
        .expect("staged snapshot");
    assert_eq!(live_paths_in(&reloaded, staged).await?, vec!["new.parquet"]);
    Ok(())
}

async fn create_test_branch(
    catalog: &Arc<dyn Catalog>,
    table: &Table,
    branch: &str,
) -> BoxResult<Table> {
    let head = table.metadata().current_snapshot_id().expect("head");
    let tx = Transaction::new(table);
    let tx = tx
        .manage_snapshots()
        .create_branch(branch, head)
        .apply(tx)?;
    Ok(tx.commit(catalog.as_ref()).await?)
}

#[tokio::test]
async fn test_stage_only_with_commit_branch_leaves_branch_unmoved() -> TestResult {
    let (catalog, table) = setup_table(HashMap::new()).await?;
    let table = append_files_direct(&catalog, &table, vec![make_data_file(
        &table,
        "base.parquet",
        7,
    )?])
    .await?;
    let table = create_test_branch(&catalog, &table, "b").await?;
    let head = table.metadata().current_snapshot_id().expect("head");
    let new_json = data_file_json(&table, make_data_file(&table, "new.parquet", 42)?)?;
    let batches = run_commit_exec(
        &table,
        &catalog,
        vec![new_json],
        InsertOp::Append,
        true,
        Some("b".to_string()),
        HashMap::new(),
    )
    .await?;
    assert_count(&batches, 42);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "t"])?)
        .await?;
    assert_eq!(reloaded.metadata().snapshots().len(), 2);
    assert_eq!(reloaded.metadata().current_snapshot_id(), Some(head));
    assert_eq!(
        reloaded
            .metadata()
            .snapshot_for_ref("b")
            .map(|s| s.snapshot_id()),
        Some(head)
    );
    Ok(())
}
