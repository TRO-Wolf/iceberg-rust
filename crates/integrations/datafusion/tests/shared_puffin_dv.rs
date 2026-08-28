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

//! F-17: DataFusion V3 DELETE/UPDATE must close a shared Puffin as one container.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{
    ArrayRef, Int32Array, Int64Array, RunArray, StringArray, UInt64Array,
};
use datafusion::arrow::datatypes::Int32Type;
use datafusion::execution::context::SessionContext;
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, ManifestContentType, NestedField, PartitionKey,
    PrimitiveType, Schema, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::deletion_vector_writer::DVFileWriter;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

const NS: &str = "f17_shared";
const TBL: &str = "items";

struct Harness {
    ctx: SessionContext,
    catalog: Arc<MemoryCatalog>,
    _warehouse: TempDir,
}

async fn harness() -> Harness {
    let warehouse = TempDir::new().expect("temp warehouse");
    let warehouse_path = warehouse
        .path()
        .to_str()
        .expect("utf8 warehouse")
        .to_string();
    let iceberg_catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path)]),
        )
        .await
        .expect("memory catalog");
    let namespace = NamespaceIdent::new(NS.to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "category", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(3, "category", Transform::Identity)
        .expect("identity(category)")
        .build();
    iceberg_catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name(TBL.to_string())
                .location(format!(
                    "{}/{}",
                    warehouse.path().to_str().expect("utf8"),
                    TBL
                ))
                .schema(schema)
                .partition_spec(partition_spec)
                .format_version(FormatVersion::V3)
                .properties(HashMap::from([
                    ("write.delete.mode".to_string(), "merge-on-read".to_string()),
                    ("write.update.mode".to_string(), "merge-on-read".to_string()),
                ]))
                .build(),
        )
        .await
        .expect("create table");

    let catalog = Arc::new(iceberg_catalog);
    let provider = IcebergCatalogProvider::try_new(catalog.clone())
        .await
        .expect("provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    Harness {
        ctx,
        catalog,
        _warehouse: warehouse,
    }
}

async fn run_sql(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

async fn sql_count(ctx: &SessionContext, sql: &str) -> u64 {
    let batches = ctx
        .sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("count column")
        .value(0)
}

async fn live_ids(ctx: &SessionContext) -> Vec<i32> {
    let batches = ctx
        .sql(&format!("SELECT id FROM catalog.{NS}.{TBL} ORDER BY id"))
        .await
        .expect("select ids")
        .collect()
        .await
        .expect("collect ids");
    let mut ids = Vec::new();
    for batch in &batches {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id int");
        ids.extend((0..batch.num_rows()).map(|row| column.value(row)));
    }
    ids
}

async fn load_table(catalog: &MemoryCatalog) -> Table {
    catalog
        .load_table(&TableIdent::new(
            NamespaceIdent::new(NS.to_string()),
            TBL.to_string(),
        ))
        .await
        .expect("load table")
}

fn decode_file_path(col: &ArrayRef, row: usize) -> String {
    if let Some(plain) = col.as_any().downcast_ref::<StringArray>() {
        return plain.value(row).to_string();
    }
    if let Some(run) = col.as_any().downcast_ref::<RunArray<Int32Type>>() {
        let values = run
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_file REE values utf8");
        return values.value(run.get_physical_index(row)).to_string();
    }
    panic!("unexpected _file column type: {:?}", col.data_type());
}

struct RowPos {
    id: i32,
    file: String,
    pos: i64,
}

async fn row_positions(table: &Table) -> Vec<RowPos> {
    let mut stream = table
        .scan()
        .select([
            "id".to_string(),
            RESERVED_COL_NAME_FILE.to_string(),
            RESERVED_COL_NAME_POS.to_string(),
        ])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("arrow");
    let mut out = Vec::new();
    while let Some(batch) = stream.try_next().await.expect("batch") {
        let ids = batch
            .column_by_name("id")
            .expect("id")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id i32");
        let file_col = batch.column_by_name(RESERVED_COL_NAME_FILE).expect("_file");
        let pos = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .expect("_pos")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("_pos i64");
        for row in 0..batch.num_rows() {
            out.push(RowPos {
                id: ids.value(row),
                file: decode_file_path(file_col, row),
                pos: pos.value(row),
            });
        }
    }
    out
}

async fn live_data_files(table: &Table) -> Vec<DataFile> {
    let mut files = Vec::new();
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("data manifest");
        for entry in manifest.entries() {
            if entry.is_alive() {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

async fn live_delete_files(table: &Table) -> Vec<DataFile> {
    let mut files = Vec::new();
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("delete manifest");
        for entry in manifest.entries() {
            if entry.is_alive() {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

async fn commit_shared_puffin(catalog: &MemoryCatalog, table: &Table, deletes: &[(String, u64)]) {
    let data_files = live_data_files(table).await;
    let by_path: HashMap<String, DataFile> = data_files
        .into_iter()
        .map(|file| (file.file_path().to_string(), file))
        .collect();
    let puffin = format!(
        "{}/data/shared-dv-{}.puffin",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output = table.file_io().new_output(&puffin).expect("puffin output");
    let mut writer = DVFileWriter::new(output);
    let schema = table.metadata().current_schema().clone();
    for (path, position) in deletes {
        let data_file = by_path
            .get(path)
            .unwrap_or_else(|| panic!("live data file {path}"));
        let spec = table
            .metadata()
            .partition_spec_by_id(data_file.partition_spec_id())
            .expect("spec")
            .as_ref()
            .clone();
        let key = PartitionKey::new(spec, schema.clone(), data_file.partition().clone())
            .expect("partition key");
        writer
            .delete(path, *position, Some(&key))
            .expect("record shared-puffin position");
    }
    let files = writer.close().await.expect("close shared puffin");
    assert!(
        !files.is_empty(),
        "Puffin must carry at least one blob, got {}",
        files.len()
    );
    if files.len() >= 2 {
        let paths: Vec<_> = files.iter().map(|file| file.file_path()).collect();
        assert!(
            paths.windows(2).all(|window| window[0] == window[1]),
            "blobs must share one Puffin path, got {paths:?}"
        );
    }
    let tx = Transaction::new(table);
    tx.row_delta()
        .add_deletes(files)
        .apply(tx)
        .expect("apply shared dv")
        .commit(catalog)
        .await
        .expect("commit shared dv");
}

async fn seed_two_file_shared_puffin(harness: &Harness) -> (String, String) {
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
    let two = rows
        .iter()
        .find(|row| row.id == 2)
        .expect("id 2 after insert");
    let five = rows
        .iter()
        .find(|row| row.id == 5)
        .expect("id 5 after insert");
    assert_ne!(
        two.file, five.file,
        "id 2 and id 5 must live in different files"
    );
    let two_file = two.file.clone();
    let five_file = five.file.clone();
    let two_pos = u64::try_from(two.pos).expect("pos 2");
    let five_pos = u64::try_from(five.pos).expect("pos 5");
    commit_shared_puffin(&harness.catalog, &table, &[
        (two_file.clone(), two_pos),
        (five_file.clone(), five_pos),
    ])
    .await;
    (two_file, five_file)
}

/// C-015 / T1 / T9: DELETE id=1 on a two-file shared Puffin must not resurrect id=5.
#[tokio::test]
async fn delete_of_one_file_must_not_resurrect_shared_puffin_sibling() {
    let harness = harness().await;
    let (_a, b) = seed_two_file_shared_puffin(&harness).await;
    assert_eq!(
        live_ids(&harness.ctx).await,
        vec![1, 3, 4, 6],
        "shared Puffin hides id 2 and id 5"
    );

    let deleted = sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"),
    )
    .await;
    assert_eq!(deleted, 1, "exactly one row matches id = 1");

    let ids = live_ids(&harness.ctx).await;
    assert_eq!(
        ids,
        vec![3, 4, 6],
        "id 5 must stay deleted (shared-Puffin sibling). live={ids:?}"
    );

    let table = load_table(&harness.catalog).await;
    let deletes = live_delete_files(&table).await;
    assert!(
        deletes
            .iter()
            .all(|file| file.file_format() == DataFileFormat::Puffin),
        "V3 forbids new position-delete files"
    );
    let referenced: Vec<_> = deletes
        .iter()
        .filter_map(|file| file.referenced_data_file())
        .collect();
    assert_eq!(
        referenced.len(),
        2,
        "both blobs must stay live after the DELETE, got {referenced:?}"
    );
    let table_after = load_table(&harness.catalog).await;
    let data_files = live_data_files(&table_after).await;
    for delete in &deletes {
        let referenced = delete
            .referenced_data_file()
            .expect("DV names its data file");
        let data = data_files
            .iter()
            .find(|file| file.file_path() == referenced)
            .unwrap_or_else(|| panic!("live data file {referenced}"));
        assert_eq!(
            delete.partition_spec_id(),
            data.partition_spec_id(),
            "T10: replacement DV keeps the data file spec"
        );
        assert_eq!(
            delete.partition(),
            data.partition(),
            "T10: replacement DV keeps the data file partition"
        );
        assert!(
            delete.content_offset().is_some() && delete.content_size_in_bytes().is_some(),
            "T10: replacement DV has blob coordinates"
        );
    }
    let sibling = deletes
        .iter()
        .find(|file| file.referenced_data_file().as_deref() == Some(b.as_str()))
        .expect("sibling DV for books");
    assert_eq!(
        sibling.record_count(),
        1,
        "T8: untouched sibling cardinality stays 1"
    );
}

/// T2: UPDATE one file in a shared Puffin. The sibling stays deleted; the updated row is live.
#[tokio::test]
async fn update_of_one_file_must_not_resurrect_shared_puffin_sibling() {
    let harness = harness().await;
    seed_two_file_shared_puffin(&harness).await;

    let updated = sql_count(
        &harness.ctx,
        &format!("UPDATE catalog.{NS}.{TBL} SET data = 'z' WHERE id = 1"),
    )
    .await;
    assert_eq!(updated, 1, "exactly one row matches id = 1");

    let ids = live_ids(&harness.ctx).await;
    assert_eq!(
        ids,
        vec![1, 3, 4, 6],
        "id 5 must stay deleted after UPDATE. live={ids:?}"
    );
    let batches = harness
        .ctx
        .sql(&format!("SELECT data FROM catalog.{NS}.{TBL} WHERE id = 1"))
        .await
        .expect("select updated")
        .collect()
        .await
        .expect("collect updated");
    let data = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("data utf8")
        .value(0);
    assert_eq!(data, "z", "updated row must carry the new value");
}

/// T12: a no-match DELETE must not write a Puffin or bump the snapshot.
#[tokio::test]
async fn no_match_delete_is_a_snapshot_noop() {
    let harness = harness().await;
    seed_two_file_shared_puffin(&harness).await;
    let before = load_table(&harness.catalog).await;
    let snapshot_before = before
        .metadata()
        .current_snapshot_id()
        .expect("snapshot before");
    let deletes_before = live_delete_files(&before).await.len();

    let deleted = sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = -1"),
    )
    .await;
    assert_eq!(deleted, 0);

    let after = load_table(&harness.catalog).await;
    assert_eq!(
        after
            .metadata()
            .current_snapshot_id()
            .expect("snapshot after"),
        snapshot_before
    );
    assert_eq!(live_delete_files(&after).await.len(), deletes_before);
    assert_eq!(live_ids(&harness.ctx).await, vec![1, 3, 4, 6]);
}

/// T4: two Puffins; touching one container leaves the other Puffin path unchanged.
#[tokio::test]
async fn delete_in_one_puffin_does_not_rewrite_the_other() {
    let harness = harness().await;
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
    let books_puffin = live_delete_files(&before)
        .await
        .into_iter()
        .find(|file| file.referenced_data_file().as_deref() == Some(five.file.as_str()))
        .expect("books DV")
        .file_path()
        .to_string();
    sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"),
    )
    .await;
    assert_eq!(live_ids(&harness.ctx).await, vec![3, 4, 6]);
    let after = load_table(&harness.catalog).await;
    let books_after = live_delete_files(&after)
        .await
        .into_iter()
        .find(|file| file.referenced_data_file().as_deref() == Some(five.file.as_str()))
        .expect("books DV after");
    assert_eq!(
        books_after.file_path(),
        books_puffin,
        "untouched Puffin path must not be rewritten"
    );
}

/// T3: touching both referenced files still leaves one live DV per file.
#[tokio::test]
async fn delete_touching_both_files_keeps_one_dv_each() {
    let harness = harness().await;
    seed_two_file_shared_puffin(&harness).await;
    let deleted = sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1 OR id = 4"),
    )
    .await;
    assert_eq!(deleted, 2);
    assert_eq!(live_ids(&harness.ctx).await, vec![3, 6]);
    let table = load_table(&harness.catalog).await;
    let deletes = live_delete_files(&table).await;
    assert_eq!(deletes.len(), 2, "one live DV per referenced file");
}

/// T13: an untouched sibling keeps its original data sequence.
#[tokio::test]
async fn untouched_sibling_keeps_original_data_sequence() {
    let harness = harness().await;
    let (_a, b) = seed_two_file_shared_puffin(&harness).await;
    let before = load_table(&harness.catalog).await;
    let seq_before = delete_data_sequence(&before, &b).await;
    sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"),
    )
    .await;
    let after = load_table(&harness.catalog).await;
    let seq_after = delete_data_sequence(&after, &b).await;
    assert_eq!(
        seq_after, seq_before,
        "sibling data sequence must not inherit the DELETE snapshot"
    );
}

/// T17: concurrent DeleteFiles of untouched sibling B rejects the frozen DELETE.
#[tokio::test]
async fn delete_rejects_concurrent_delete_of_untouched_sibling() {
    let harness = harness().await;
    let (_a, b) = seed_two_file_shared_puffin(&harness).await;
    let plan = harness
        .ctx
        .sql(&format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"))
        .await
        .expect("logical")
        .create_physical_plan()
        .await
        .expect("frozen physical plan");

    let table = load_table(&harness.catalog).await;
    let tx = Transaction::new(&table);
    tx.delete_files()
        .delete_files([b.clone()])
        .apply(tx)
        .expect("apply delete_files")
        .commit(harness.catalog.as_ref())
        .await
        .expect("concurrent DeleteFiles of sibling B");

    let err = datafusion::physical_plan::collect(plan, harness.ctx.task_ctx())
        .await
        .expect_err("DELETE must reject concurrent Delete of sibling B");
    let message = err.to_string();
    assert!(
        message.contains("missing data files") || message.contains("conflicting delete"),
        "expected files-exist rejection of sibling B, got {message}"
    );
}

/// T18: concurrent DeleteFiles of touched file A also rejects the frozen DELETE.
#[tokio::test]
async fn delete_rejects_concurrent_delete_of_touched_file() {
    let harness = harness().await;
    let (a, _b) = seed_two_file_shared_puffin(&harness).await;
    let plan = harness
        .ctx
        .sql(&format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"))
        .await
        .expect("logical")
        .create_physical_plan()
        .await
        .expect("frozen physical plan");
    let table = load_table(&harness.catalog).await;
    let tx = Transaction::new(&table);
    tx.delete_files()
        .delete_files([a.clone()])
        .apply(tx)
        .expect("apply delete_files")
        .commit(harness.catalog.as_ref())
        .await
        .expect("concurrent DeleteFiles of A");
    let err = datafusion::physical_plan::collect(plan, harness.ctx.task_ctx())
        .await
        .expect_err("DELETE must reject concurrent Delete of touched A");
    let message = err.to_string();
    assert!(
        message.contains("missing data files") || message.contains("conflicting delete"),
        "expected files-exist rejection of A, got {message}"
    );
}

/// T23: concurrent DeleteFiles of an unrelated file is outside the replacement set.
#[tokio::test]
async fn delete_allows_concurrent_delete_of_unrelated_file() {
    let harness = harness().await;
    seed_two_file_shared_puffin(&harness).await;
    run_sql(
        &harness.ctx,
        &format!("INSERT INTO catalog.{NS}.{TBL} VALUES (7, 'g', 'toys')"),
    )
    .await;
    let table = load_table(&harness.catalog).await;
    let toys = live_data_files(&table)
        .await
        .into_iter()
        .find(|file| {
            !file.file_path().contains("electronics") && !file.file_path().contains("books")
        })
        .expect("toys data file")
        .file_path()
        .to_string();
    let plan = harness
        .ctx
        .sql(&format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"))
        .await
        .expect("logical")
        .create_physical_plan()
        .await
        .expect("frozen physical plan");
    let table = load_table(&harness.catalog).await;
    let tx = Transaction::new(&table);
    tx.delete_files()
        .delete_files([toys])
        .apply(tx)
        .expect("apply delete_files")
        .commit(harness.catalog.as_ref())
        .await
        .expect("concurrent DeleteFiles of unrelated C");
    datafusion::physical_plan::collect(plan, harness.ctx.task_ctx())
        .await
        .expect("DELETE must succeed when only unrelated C was removed");
    assert_eq!(live_ids(&harness.ctx).await, vec![3, 4, 6]);
}

async fn delete_data_sequence(table: &Table, referenced: &str) -> i64 {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("delete manifest");
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let file = entry.data_file();
            if file.referenced_data_file().as_deref() == Some(referenced) {
                return entry.sequence_number().expect("dv data sequence");
            }
        }
    }
    panic!("no live DV for {referenced}");
}
