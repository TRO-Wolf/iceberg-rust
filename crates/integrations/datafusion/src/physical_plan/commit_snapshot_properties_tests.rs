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
use iceberg::transaction::staged_snapshot_for_wap_id;
use iceberg::{NamespaceIdent, TableIdent};

use super::tests::{
    BoxResult, TestResult, append_files_direct, assert_count, data_file_json, make_data_file,
    run_commit_exec, setup_table,
};
use super::{IcebergCommitExec, OPERATION_ID_PROP};
use crate::table::IcebergTableProvider;

async fn commit_snapshot_properties_of(
    provider: &IcebergTableProvider,
    state: &dyn datafusion::catalog::Session,
) -> BoxResult<HashMap<String, String>> {
    use datafusion::datasource::TableProvider;
    use datafusion::physical_plan::empty::EmptyExec;
    let input = Arc::new(EmptyExec::new(provider.schema())) as Arc<dyn ExecutionPlan>;
    let plan = provider.insert_into(state, input, InsertOp::Append).await?;
    Ok(plan
        .downcast_ref::<IcebergCommitExec>()
        .expect("insert_into plans IcebergCommitExec")
        .snapshot_properties
        .clone())
}

#[tokio::test]
async fn test_provider_snapshot_properties_default_empty_and_threads() -> TestResult {
    let (catalog, _) = setup_table(HashMap::new()).await?;
    let ctx = SessionContext::new();
    let state = ctx.state();
    let default = IcebergTableProvider::try_new(
        Arc::clone(&catalog),
        NamespaceIdent::new("ns".to_string()),
        "t".to_string(),
    )
    .await?;
    assert!(
        commit_snapshot_properties_of(&default, &state)
            .await?
            .is_empty()
    );
    let stamped = IcebergTableProvider::try_new(
        Arc::clone(&catalog),
        NamespaceIdent::new("ns".to_string()),
        "t".to_string(),
    )
    .await?
    .with_snapshot_properties(HashMap::from([("k".to_string(), "v".to_string())]));
    assert_eq!(
        commit_snapshot_properties_of(&stamped, &state).await?,
        HashMap::from([("k".to_string(), "v".to_string())])
    );
    Ok(())
}

#[tokio::test]
async fn test_default_commit_stamps_operation_id_without_caller_keys() -> TestResult {
    let (catalog, table) = setup_table(HashMap::new()).await?;
    let new_json = data_file_json(&table, make_data_file(&table, "new.parquet", 42)?)?;
    let batches = run_commit_exec(
        &table,
        &catalog,
        vec![new_json],
        InsertOp::Append,
        false,
        None,
        HashMap::new(),
    )
    .await?;
    assert_count(&batches, 42);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "t"])?)
        .await?;
    let summary = reloaded
        .metadata()
        .current_snapshot()
        .expect("head")
        .summary();
    let operation_id = summary
        .additional_properties
        .get(OPERATION_ID_PROP)
        .expect("operation id stamp");
    assert!(!operation_id.is_empty());
    assert!(!summary.additional_properties.contains_key("k"));
    Ok(())
}

#[tokio::test]
async fn test_append_merges_caller_snapshot_properties() -> TestResult {
    let (catalog, table) = setup_table(HashMap::new()).await?;
    let new_json = data_file_json(&table, make_data_file(&table, "new.parquet", 42)?)?;
    let batches = run_commit_exec(
        &table,
        &catalog,
        vec![new_json],
        InsertOp::Append,
        false,
        None,
        HashMap::from([
            ("wap.id".to_string(), "append-wap".to_string()),
            ("custom.key".to_string(), "custom-value".to_string()),
        ]),
    )
    .await?;
    assert_count(&batches, 42);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "t"])?)
        .await?;
    let summary = reloaded
        .metadata()
        .current_snapshot()
        .expect("head")
        .summary();
    assert_eq!(
        summary.additional_properties.get("wap.id"),
        Some(&"append-wap".to_string())
    );
    assert_eq!(
        summary.additional_properties.get("custom.key"),
        Some(&"custom-value".to_string())
    );
    assert!(
        !summary
            .additional_properties
            .get(OPERATION_ID_PROP)
            .expect("operation id stamp")
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn test_overwrite_merges_caller_snapshot_properties() -> TestResult {
    let (catalog, table) = setup_table(HashMap::new()).await?;
    let table = append_files_direct(&catalog, &table, vec![make_data_file(
        &table,
        "base.parquet",
        7,
    )?])
    .await?;
    let new_json = data_file_json(&table, make_data_file(&table, "new.parquet", 42)?)?;
    let batches = run_commit_exec(
        &table,
        &catalog,
        vec![new_json],
        InsertOp::Overwrite,
        false,
        None,
        HashMap::from([
            ("wap.id".to_string(), "overwrite-wap".to_string()),
            ("custom.key".to_string(), "custom-value".to_string()),
        ]),
    )
    .await?;
    assert_count(&batches, 42);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "t"])?)
        .await?;
    let summary = reloaded
        .metadata()
        .current_snapshot()
        .expect("head")
        .summary();
    assert_eq!(
        summary.additional_properties.get("wap.id"),
        Some(&"overwrite-wap".to_string())
    );
    assert_eq!(
        summary.additional_properties.get("custom.key"),
        Some(&"custom-value".to_string())
    );
    assert!(
        !summary
            .additional_properties
            .get(OPERATION_ID_PROP)
            .expect("operation id stamp")
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn test_append_caller_operation_id_does_not_win() -> TestResult {
    let (catalog, table) = setup_table(HashMap::new()).await?;
    let new_json = data_file_json(&table, make_data_file(&table, "new.parquet", 42)?)?;
    let batches = run_commit_exec(
        &table,
        &catalog,
        vec![new_json],
        InsertOp::Append,
        false,
        None,
        HashMap::from([
            (OPERATION_ID_PROP.to_string(), "forged".to_string()),
            ("k".to_string(), "v".to_string()),
        ]),
    )
    .await?;
    assert_count(&batches, 42);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "t"])?)
        .await?;
    let summary = reloaded
        .metadata()
        .current_snapshot()
        .expect("head")
        .summary();
    let operation_id = summary
        .additional_properties
        .get(OPERATION_ID_PROP)
        .expect("operation id stamp");
    assert_ne!(operation_id, "forged");
    assert!(!operation_id.is_empty());
    assert_eq!(
        summary.additional_properties.get("k"),
        Some(&"v".to_string())
    );
    Ok(())
}

#[tokio::test]
async fn test_overwrite_caller_operation_id_does_not_win() -> TestResult {
    let (catalog, table) = setup_table(HashMap::new()).await?;
    let table = append_files_direct(&catalog, &table, vec![make_data_file(
        &table,
        "base.parquet",
        7,
    )?])
    .await?;
    let new_json = data_file_json(&table, make_data_file(&table, "new.parquet", 42)?)?;
    let batches = run_commit_exec(
        &table,
        &catalog,
        vec![new_json],
        InsertOp::Overwrite,
        false,
        None,
        HashMap::from([
            (OPERATION_ID_PROP.to_string(), "forged".to_string()),
            ("k".to_string(), "v".to_string()),
        ]),
    )
    .await?;
    assert_count(&batches, 42);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "t"])?)
        .await?;
    let summary = reloaded
        .metadata()
        .current_snapshot()
        .expect("head")
        .summary();
    let operation_id = summary
        .additional_properties
        .get(OPERATION_ID_PROP)
        .expect("operation id stamp");
    assert_ne!(operation_id, "forged");
    assert!(!operation_id.is_empty());
    assert_eq!(
        summary.additional_properties.get("k"),
        Some(&"v".to_string())
    );
    Ok(())
}

#[tokio::test]
async fn test_stage_only_snapshot_properties_land_on_staged_snapshot() -> TestResult {
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
        HashMap::from([("wap.id".to_string(), "staged-wap".to_string())]),
    )
    .await?;
    assert_count(&batches, 42);
    let reloaded = catalog
        .load_table(&TableIdent::from_strs(["ns", "t"])?)
        .await?;
    assert_eq!(reloaded.metadata().current_snapshot_id(), Some(head));
    let staged = staged_snapshot_for_wap_id(reloaded.metadata(), "staged-wap")?;
    assert_ne!(staged.snapshot_id(), head);
    assert_eq!(
        staged.summary().additional_properties.get("wap.id"),
        Some(&"staged-wap".to_string())
    );
    assert!(
        !staged
            .summary()
            .additional_properties
            .get(OPERATION_ID_PROP)
            .expect("operation id stamp")
            .is_empty()
    );
    Ok(())
}
