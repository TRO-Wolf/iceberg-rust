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
    DataFile, FormatVersion, ManifestContentType, NestedField, PartitionKey, PrimitiveType, Schema,
    Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::deletion_vector_writer::DVFileWriter;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

pub(crate) const NS: &str = "f17_shared";
pub(crate) const TBL: &str = "items";

static SUITE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub(crate) struct Harness {
    pub(crate) ctx: SessionContext,
    pub(crate) catalog: Arc<MemoryCatalog>,
    pub(crate) _warehouse: TempDir,
    /// Held for the whole test so T19/T21's process env injector cannot leak.
    _suite: tokio::sync::MutexGuard<'static, ()>,
}

pub(crate) async fn harness() -> Harness {
    harness_with(FormatVersion::V3).await
}

pub(crate) async fn harness_with(format_version: FormatVersion) -> Harness {
    let suite = SUITE_LOCK.lock().await;
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
                .format_version(format_version)
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
        _suite: suite,
    }
}

pub(crate) async fn run_sql(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

pub(crate) async fn sql_count(ctx: &SessionContext, sql: &str) -> u64 {
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

pub(crate) async fn live_ids(ctx: &SessionContext) -> Vec<i32> {
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

pub(crate) async fn load_table(catalog: &MemoryCatalog) -> Table {
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

pub(crate) struct RowPos {
    pub(crate) id: i32,
    pub(crate) file: String,
    pub(crate) pos: i64,
}

pub(crate) async fn row_positions(table: &Table) -> Vec<RowPos> {
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

pub(crate) async fn live_data_files(table: &Table) -> Vec<DataFile> {
    collect_alive(table, ManifestContentType::Data).await
}

pub(crate) async fn live_delete_files(table: &Table) -> Vec<DataFile> {
    collect_alive(table, ManifestContentType::Deletes).await
}

async fn collect_alive(table: &Table, content: ManifestContentType) -> Vec<DataFile> {
    let mut files = Vec::new();
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != content {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

pub(crate) async fn commit_shared_puffin(
    catalog: &MemoryCatalog,
    table: &Table,
    deletes: &[(String, u64)],
) {
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
    let mut writer = DVFileWriter::new(output).unpartitioned();
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

pub(crate) async fn seed_two_file_shared_puffin(harness: &Harness) -> (String, String) {
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

pub(crate) async fn delete_data_sequence(table: &Table, referenced: &str) -> i64 {
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

pub(crate) async fn snapshot_id(catalog: &MemoryCatalog) -> i64 {
    load_table(catalog)
        .await
        .metadata()
        .current_snapshot_id()
        .expect("snapshot")
}

pub(crate) fn list_table_files(table: &Table) -> Vec<String> {
    let mut out = Vec::new();
    walk_files(std::path::Path::new(table.metadata().location()), &mut out);
    out.sort();
    out
}

fn walk_files(path: &std::path::Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let child = entry.path();
        if child.is_dir() {
            walk_files(&child, out);
        } else if let Some(text) = child.to_str() {
            out.push(text.to_string());
        }
    }
}
