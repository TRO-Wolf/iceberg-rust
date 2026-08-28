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

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use datafusion::physical_plan::collect;
use iceberg::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};
use iceberg::memory::MemoryCatalog;
use iceberg::spec::{DataContentType, DataFileBuilder, DataFileFormat, PartitionKey};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use parquet::file::properties::WriterProperties;

use super::harness::{
    Harness, NS, TBL, commit_shared_puffin, harness, list_table_files, live_data_files,
    live_delete_files, live_ids, load_table, row_positions, run_sql, seed_two_file_shared_puffin,
    snapshot_id, sql_count,
};

struct FailCloseGuard;
impl Drop for FailCloseGuard {
    fn drop(&mut self) {
        // SAFETY: Harness holds SUITE_LOCK for this test, so no sibling test is in close.
        unsafe {
            std::env::remove_var("ICEBERG_FAIL_DV_CONTAINER_BEFORE_WRITE");
        }
    }
}

fn arm_fail_before_container_write() -> FailCloseGuard {
    // SAFETY: Harness holds SUITE_LOCK; this test is the only close() in the process.
    unsafe {
        std::env::set_var("ICEBERG_FAIL_DV_CONTAINER_BEFORE_WRITE", "1");
    }
    FailCloseGuard
}

/// T5: two Puffins; touching both containers rewrites each independently.
#[tokio::test]
async fn delete_touching_both_puffins_rewrites_each_container() {
    let harness = harness().await;
    let (electronics, books, electronics_puffin, books_puffin) =
        seed_two_separate_puffins(&harness).await;
    let deleted = sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1 OR id = 4"),
    )
    .await;
    assert_eq!(deleted, 2);
    assert_eq!(live_ids(&harness.ctx).await, vec![3, 6]);
    let after = load_table(&harness.catalog).await;
    let deletes = live_delete_files(&after).await;
    assert_eq!(deletes.len(), 2, "one live DV per referenced file");
    let electronics_after = deletes
        .iter()
        .find(|file| file.referenced_data_file().as_deref() == Some(electronics.as_str()))
        .expect("electronics DV");
    let books_after = deletes
        .iter()
        .find(|file| file.referenced_data_file().as_deref() == Some(books.as_str()))
        .expect("books DV");
    assert_ne!(
        electronics_after.file_path(),
        electronics_puffin,
        "electronics Puffin must close independently"
    );
    assert_ne!(
        books_after.file_path(),
        books_puffin,
        "books Puffin must close independently"
    );
    assert_ne!(
        electronics_after.file_path(),
        books_after.file_path(),
        "containers must not merge"
    );
}

/// T11: an equality delete stays live and still applies after shared-Puffin DELETE.
#[tokio::test]
async fn equality_delete_survives_shared_puffin_delete() {
    let harness = harness().await;
    seed_two_file_shared_puffin(&harness).await;
    let table = load_table(&harness.catalog).await;
    commit_equality_delete(&harness.catalog, &table, 3).await;
    assert_eq!(live_ids(&harness.ctx).await, vec![1, 4, 6]);
    sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"),
    )
    .await;
    assert_eq!(live_ids(&harness.ctx).await, vec![4, 6]);
    let after = load_table(&harness.catalog).await;
    let eq_live = live_delete_files(&after)
        .await
        .into_iter()
        .any(|file| file.content_type() == DataContentType::EqualityDeletes);
    assert!(eq_live, "equality delete must remain live");
}

/// T14: concurrent Replace of untouched B rejects the frozen DELETE.
#[tokio::test]
async fn delete_rejects_concurrent_replace_of_untouched_sibling() {
    let harness = harness().await;
    let (_a, b) = seed_two_file_shared_puffin(&harness).await;
    set_snapshot_isolation(&harness).await;
    let plan = freeze_sql(
        &harness,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"),
    )
    .await;
    let table = load_table(&harness.catalog).await;
    rewrite_data_file(&harness.catalog, &table, &b).await;
    let err = collect(plan, harness.ctx.task_ctx())
        .await
        .expect_err("DELETE must reject concurrent Replace of sibling B");
    assert!(
        err.to_string().contains("missing data files"),
        "files-exist must cover B, got {err}"
    );
    assert!(
        live_ids(&harness.ctx).await.contains(&1),
        "rejected DELETE must not remove id 1"
    );
}

/// T15: concurrent Replace of B rejects the frozen UPDATE.
#[tokio::test]
async fn update_rejects_concurrent_replace_of_untouched_sibling() {
    let harness = harness().await;
    let (_a, b) = seed_two_file_shared_puffin(&harness).await;
    set_snapshot_isolation(&harness).await;
    let plan = freeze_sql(
        &harness,
        &format!("UPDATE catalog.{NS}.{TBL} SET data = 'z' WHERE id = 1"),
    )
    .await;
    let table = load_table(&harness.catalog).await;
    rewrite_data_file(&harness.catalog, &table, &b).await;
    let err = collect(plan, harness.ctx.task_ctx())
        .await
        .expect_err("UPDATE must reject concurrent Replace of sibling B");
    assert!(
        err.to_string().contains("missing data files"),
        "files-exist must cover B, got {err}"
    );
    let data = harness
        .ctx
        .sql(&format!("SELECT data FROM catalog.{NS}.{TBL} WHERE id = 1"))
        .await
        .expect("select")
        .collect()
        .await
        .expect("collect");
    let value = data[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("data utf8")
        .value(0);
    assert_eq!(value, "a", "rejected UPDATE must not change id 1");
}

/// T16: concurrent DeleteFiles of B rejects the frozen UPDATE.
#[tokio::test]
async fn update_rejects_concurrent_delete_of_untouched_sibling() {
    let harness = harness().await;
    let (_a, b) = seed_two_file_shared_puffin(&harness).await;
    let plan = freeze_sql(
        &harness,
        &format!("UPDATE catalog.{NS}.{TBL} SET data = 'z' WHERE id = 1"),
    )
    .await;
    let table = load_table(&harness.catalog).await;
    let tx = Transaction::new(&table);
    tx.delete_files()
        .delete_files([b])
        .apply(tx)
        .expect("apply delete_files")
        .commit(harness.catalog.as_ref())
        .await
        .expect("concurrent DeleteFiles of B");
    let err = collect(plan, harness.ctx.task_ctx())
        .await
        .expect_err("UPDATE must reject concurrent Delete of sibling B");
    let message = err.to_string();
    assert!(
        message.contains("missing data files") || message.contains("conflicting delete"),
        "expected deleted-files rejection of B, got {message}"
    );
}

/// T19: a shared-Puffin close failure before the replacement write is a byte-noop.
#[tokio::test]
async fn delete_pre_output_failure_is_a_byte_noop() {
    let harness = harness().await;
    seed_two_file_shared_puffin(&harness).await;
    let table = load_table(&harness.catalog).await;
    let snap_before = snapshot_id(&harness.catalog).await;
    let files_before = list_table_files(&table);
    let _arm = arm_fail_before_container_write();
    let err = harness
        .ctx
        .sql(&format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"))
        .await
        .expect("plan")
        .collect()
        .await
        .expect_err("DELETE must fail before the replacement Puffin is written");
    assert!(
        err.to_string()
            .contains("injected failure before shared-Puffin replacement write"),
        "container-close precondition, got {err}"
    );
    drop(_arm);
    let table = load_table(&harness.catalog).await;
    assert_eq!(snapshot_id(&harness.catalog).await, snap_before);
    assert_eq!(list_table_files(&table), files_before);
}

/// T20: a rejected DELETE after Puffin close leaves exactly that Puffin unreferenced.
#[tokio::test]
async fn delete_post_output_commit_failure_leaves_one_orphan_puffin() {
    let harness = harness().await;
    let (_a, b) = seed_two_file_shared_puffin(&harness).await;
    set_snapshot_isolation(&harness).await;
    let plan = freeze_sql(
        &harness,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"),
    )
    .await;
    let table = load_table(&harness.catalog).await;
    rewrite_data_file(&harness.catalog, &table, &b).await;
    let table = load_table(&harness.catalog).await;
    let snap_before = snapshot_id(&harness.catalog).await;
    let puffins_before = puffin_paths(&list_table_files(&table));
    collect(plan, harness.ctx.task_ctx())
        .await
        .expect_err("DELETE commit must fail after the Puffin is written");
    let table = load_table(&harness.catalog).await;
    assert_eq!(snapshot_id(&harness.catalog).await, snap_before);
    let puffins_after = puffin_paths(&list_table_files(&table));
    let new_puffins: Vec<_> = puffins_after.difference(&puffins_before).cloned().collect();
    assert_eq!(
        new_puffins.len(),
        1,
        "exactly one replacement Puffin, got {new_puffins:?}"
    );
    let live = live_delete_files(&table).await;
    assert!(
        live.iter().all(|file| file.file_path() != new_puffins[0]),
        "the new Puffin must be unreferenced"
    );
}

/// T21: UPDATE fails before the Puffin opens; one unreferenced data file, no Puffin.
#[tokio::test]
async fn update_pre_puffin_failure_leaves_one_orphan_data_file() {
    let harness = harness().await;
    seed_two_file_shared_puffin(&harness).await;
    let table = load_table(&harness.catalog).await;
    let snap_before = snapshot_id(&harness.catalog).await;
    let files_before: HashSet<_> = list_table_files(&table).into_iter().collect();
    let _arm = arm_fail_before_container_write();
    let err = harness
        .ctx
        .sql(&format!(
            "UPDATE catalog.{NS}.{TBL} SET data = 'z' WHERE id = 1"
        ))
        .await
        .expect("plan")
        .collect()
        .await
        .expect_err("UPDATE must fail before the replacement Puffin is written");
    assert!(
        err.to_string()
            .contains("injected failure before shared-Puffin replacement write"),
        "container-close precondition, got {err}"
    );
    drop(_arm);
    let table = load_table(&harness.catalog).await;
    assert_eq!(snapshot_id(&harness.catalog).await, snap_before);
    let files_after: HashSet<_> = list_table_files(&table).into_iter().collect();
    let new_files: Vec<_> = files_after.difference(&files_before).cloned().collect();
    let new_puffins: Vec<_> = new_files
        .iter()
        .filter(|path| path.ends_with(".puffin"))
        .collect();
    let new_parquet: Vec<_> = new_files
        .iter()
        .filter(|path| path.ends_with(".parquet"))
        .collect();
    assert!(
        new_puffins.is_empty(),
        "no Puffin before the writer opens, got {new_puffins:?}"
    );
    assert_eq!(
        new_parquet.len(),
        1,
        "exactly one replacement data file, got {new_parquet:?}"
    );
}

/// T22: a rejected UPDATE after both writers close leaves one data file and one Puffin.
#[tokio::test]
async fn update_post_output_commit_failure_leaves_data_and_puffin_orphans() {
    let harness = harness().await;
    let (_a, b) = seed_two_file_shared_puffin(&harness).await;
    set_snapshot_isolation(&harness).await;
    let plan = freeze_sql(
        &harness,
        &format!("UPDATE catalog.{NS}.{TBL} SET data = 'z' WHERE id = 1"),
    )
    .await;
    let table = load_table(&harness.catalog).await;
    let tx = Transaction::new(&table);
    tx.delete_files()
        .delete_files([b])
        .apply(tx)
        .expect("apply delete_files")
        .commit(harness.catalog.as_ref())
        .await
        .expect("concurrent DeleteFiles of B");
    let table = load_table(&harness.catalog).await;
    let snap_before = snapshot_id(&harness.catalog).await;
    let files_before: HashSet<_> = list_table_files(&table).into_iter().collect();
    collect(plan, harness.ctx.task_ctx())
        .await
        .expect_err("UPDATE commit must fail after both writers close");
    let table = load_table(&harness.catalog).await;
    assert_eq!(snapshot_id(&harness.catalog).await, snap_before);
    let files_after: HashSet<_> = list_table_files(&table).into_iter().collect();
    let new_files: Vec<_> = files_after.difference(&files_before).cloned().collect();
    let new_puffins: Vec<_> = new_files
        .iter()
        .filter(|path| path.ends_with(".puffin"))
        .cloned()
        .collect();
    let new_parquet: Vec<_> = new_files
        .iter()
        .filter(|path| path.ends_with(".parquet"))
        .cloned()
        .collect();
    assert_eq!(
        new_puffins.len(),
        1,
        "exactly one replacement Puffin, got {new_puffins:?}"
    );
    assert_eq!(
        new_parquet.len(),
        1,
        "exactly one replacement data file, got {new_parquet:?}"
    );
}

async fn seed_two_separate_puffins(harness: &Harness) -> (String, String, String, String) {
    run_sql(
        &harness.ctx,
        &format!(
            "INSERT INTO catalog.{NS}.{TBL} VALUES \
             (1, 'a', 'electronics'), (2, 'b', 'electronics'), (3, 'c', 'electronics'), \
             (4, 'd', 'books'), (5, 'e', 'books'), (6, 'f', 'books')"
        ),
    )
    .await;
    let table = load_table(&harness.catalog).await;
    let rows = row_positions(&table).await;
    let two = rows.iter().find(|row| row.id == 2).expect("id 2");
    let five = rows.iter().find(|row| row.id == 5).expect("id 5");
    commit_shared_puffin(&harness.catalog, &table, &[(
        two.file.clone(),
        u64::try_from(two.pos).expect("pos 2"),
    )])
    .await;
    let table = load_table(&harness.catalog).await;
    commit_shared_puffin(&harness.catalog, &table, &[(
        five.file.clone(),
        u64::try_from(five.pos).expect("pos 5"),
    )])
    .await;
    let before = load_table(&harness.catalog).await;
    let deletes = live_delete_files(&before).await;
    let electronics_puffin = deletes
        .iter()
        .find(|file| file.referenced_data_file().as_deref() == Some(two.file.as_str()))
        .expect("electronics DV")
        .file_path()
        .to_string();
    let books_puffin = deletes
        .iter()
        .find(|file| file.referenced_data_file().as_deref() == Some(five.file.as_str()))
        .expect("books DV")
        .file_path()
        .to_string();
    (
        two.file.clone(),
        five.file.clone(),
        electronics_puffin,
        books_puffin,
    )
}

async fn commit_equality_delete(catalog: &MemoryCatalog, table: &Table, id: i32) {
    let schema = table.metadata().current_schema().clone();
    let config =
        EqualityDeleteWriterConfig::new(vec![1], schema.clone()).expect("eq-delete config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("eqdel-{id}"),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let delete_schema =
        Arc::new(arrow_schema_to_schema(config.projected_arrow_schema_ref()).expect("eq schema"));
    let parquet_builder =
        ParquetWriterBuilder::new(WriterProperties::builder().build(), delete_schema);
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let electronics = live_data_files(table)
        .await
        .into_iter()
        .find(|file| file.file_path().contains("electronics"))
        .expect("electronics data file");
    let spec = table
        .metadata()
        .partition_spec_by_id(electronics.partition_spec_id())
        .expect("spec")
        .as_ref()
        .clone();
    let key = PartitionKey::new(spec, schema.clone(), electronics.partition().clone())
        .expect("eq-delete partition key");
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(Some(key))
        .await
        .expect("build eq-delete writer");
    let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).expect("schema → arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int32Array::from(vec![id])) as ArrayRef,
        Arc::new(StringArray::from(vec!["c"])) as ArrayRef,
        Arc::new(StringArray::from(vec!["electronics"])) as ArrayRef,
    ])
    .expect("eq-delete batch");
    writer.write(batch).await.expect("write eq-delete");
    let files = writer.close().await.expect("close eq-delete");
    let tx = Transaction::new(table);
    tx.row_delta()
        .add_deletes(files)
        .apply(tx)
        .expect("apply eq-delete")
        .commit(catalog)
        .await
        .expect("commit eq-delete");
}

async fn set_snapshot_isolation(harness: &Harness) {
    let table = load_table(&harness.catalog).await;
    let tx = Transaction::new(&table);
    tx.update_table_properties()
        .set(
            "write.delete.isolation-level".to_string(),
            "snapshot".to_string(),
        )
        .set(
            "write.update.isolation-level".to_string(),
            "snapshot".to_string(),
        )
        .apply(tx)
        .expect("apply isolation")
        .commit(harness.catalog.as_ref())
        .await
        .expect("commit isolation");
}

async fn freeze_sql(
    harness: &Harness,
    sql: &str,
) -> Arc<dyn datafusion::physical_plan::ExecutionPlan> {
    harness
        .ctx
        .sql(sql)
        .await
        .expect("logical")
        .create_physical_plan()
        .await
        .expect("frozen physical plan")
}

async fn rewrite_data_file(catalog: &MemoryCatalog, table: &Table, path: &str) {
    let old = live_data_files(table)
        .await
        .into_iter()
        .find(|file| file.file_path() == path)
        .unwrap_or_else(|| panic!("live data file {path}"));
    let new_path = format!(
        "{}/data/rewritten-{}.parquet",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let bytes = table
        .file_io()
        .new_input(old.file_path())
        .expect("input")
        .read()
        .await
        .expect("read parquet");
    table
        .file_io()
        .new_output(&new_path)
        .expect("output")
        .write(bytes)
        .await
        .expect("copy parquet");
    let new_file = DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(new_path)
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(old.file_size_in_bytes())
        .record_count(old.record_count())
        .partition_spec_id(old.partition_spec_id())
        .partition(old.partition().clone())
        .column_sizes(old.column_sizes().clone())
        .value_counts(old.value_counts().clone())
        .null_value_counts(old.null_value_counts().clone())
        .lower_bounds(old.lower_bounds().clone())
        .upper_bounds(old.upper_bounds().clone())
        .split_offsets(old.split_offsets().map(|offsets| offsets.to_vec()))
        .first_row_id(old.first_row_id())
        .build()
        .expect("rewritten data file");
    let tx = Transaction::new(table);
    tx.rewrite_files(vec![old], vec![new_file])
        .apply(tx)
        .expect("apply rewrite")
        .commit(catalog)
        .await
        .expect("commit rewrite");
}

fn puffin_paths(files: &[String]) -> HashSet<String> {
    files
        .iter()
        .filter(|path| path.ends_with(".puffin"))
        .cloned()
        .collect()
}
