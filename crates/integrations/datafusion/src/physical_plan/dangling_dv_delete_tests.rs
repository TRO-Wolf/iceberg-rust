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
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};
use iceberg::spec::{
    DataFile, FormatVersion, ManifestContentType, NestedField, PartitionKey, PrimitiveType,
    Schema as IcebergSchema, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::deletion_vector_writer::DVFileWriter;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use tempfile::TempDir;

use crate::IcebergCatalogProvider;

struct Fixture {
    ctx: SessionContext,
    catalog: Arc<MemoryCatalog>,
    _warehouse: TempDir,
}

async fn fixture() -> Fixture {
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
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("t".to_string())
                .location(format!("{}/t", warehouse.path().to_str().expect("utf8")))
                .schema(schema)
                .format_version(FormatVersion::V3)
                .properties(HashMap::from([
                    ("write.delete.mode".to_string(), "copy-on-write".to_string()),
                    ("write.update.mode".to_string(), "copy-on-write".to_string()),
                ]))
                .build(),
        )
        .await
        .expect("table");
    let provider = IcebergCatalogProvider::try_new(catalog.clone())
        .await
        .expect("catalog provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    Fixture {
        ctx,
        catalog,
        _warehouse: warehouse,
    }
}

async fn live_ids(ctx: &SessionContext) -> Vec<i32> {
    let batches = ctx
        .sql("SELECT id FROM catalog.ns.t ORDER BY id")
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

async fn dml_count(ctx: &SessionContext, sql: &str) -> u64 {
    let batches = ctx
        .sql(sql)
        .await
        .expect("plan DML")
        .collect()
        .await
        .expect("execute DML");
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

async fn load_table(catalog: &MemoryCatalog) -> Table {
    catalog
        .load_table(&TableIdent::new(
            NamespaceIdent::new("ns".to_string()),
            "t".to_string(),
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

async fn id_row_position(table: &Table, wanted: i32) -> (String, i64) {
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
            if ids.value(row) == wanted {
                return (decode_file_path(file_col, row), pos.value(row));
            }
        }
    }
    panic!("id {wanted} not found in scan");
}

async fn live_delete_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
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

fn summary_value(table: &Table, key: &str) -> Option<String> {
    table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .summary()
        .additional_properties
        .get(key)
        .cloned()
}

#[tokio::test]
async fn test_dangling_dv_cow_delete_drops_dv_of_removed_data_file() {
    let fixture = fixture().await;
    fixture
        .ctx
        .sql("INSERT INTO catalog.ns.t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        .await
        .expect("plan first insert")
        .collect()
        .await
        .expect("first insert");
    fixture
        .ctx
        .sql("INSERT INTO catalog.ns.t VALUES (4, 'd'), (5, 'e'), (6, 'f')")
        .await
        .expect("plan second insert")
        .collect()
        .await
        .expect("second insert");
    let table = load_table(&fixture.catalog).await;
    let (dv_target_path, dv_target_pos) = id_row_position(&table, 2).await;
    let data_files = {
        let snapshot = table.metadata().current_snapshot().expect("snapshot");
        let list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .expect("manifest list");
        let mut files = Vec::new();
        for manifest_file in list.entries() {
            if manifest_file.content != ManifestContentType::Data {
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
    };
    let target = data_files
        .iter()
        .find(|file| file.file_path() == dv_target_path)
        .expect("live data file for the DV target");
    let spec = table
        .metadata()
        .partition_spec_by_id(target.partition_spec_id())
        .expect("spec")
        .as_ref()
        .clone();
    let key = PartitionKey::new(
        spec,
        table.metadata().current_schema().clone(),
        target.partition().clone(),
    )
    .expect("partition key");
    let puffin = format!(
        "{}/data/dv-{}.puffin",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output = table.file_io().new_output(&puffin).expect("puffin output");
    let mut writer = DVFileWriter::new(output).unpartitioned();
    writer
        .delete(
            &dv_target_path,
            u64::try_from(dv_target_pos).expect("pos"),
            Some(&key),
        )
        .expect("record dv");
    let dv_files = writer.close().await.expect("close dv");
    assert_eq!(dv_files.len(), 1);
    assert_eq!(
        dv_files[0].referenced_data_file().as_deref(),
        Some(dv_target_path.as_str())
    );
    let tx = Transaction::new(&table);
    let table = tx
        .row_delta()
        .add_deletes(dv_files)
        .apply(tx)
        .expect("apply dv")
        .commit(&*fixture.catalog)
        .await
        .expect("commit dv");
    assert_eq!(live_delete_files(&table).await.len(), 1);
    assert_eq!(live_ids(&fixture.ctx).await, vec![1, 3, 4, 5, 6]);

    let deleted = dml_count(&fixture.ctx, "DELETE FROM catalog.ns.t WHERE id < 4").await;
    assert_eq!(deleted, 2);
    let table = load_table(&fixture.catalog).await;
    assert!(
        live_delete_files(&table).await.is_empty(),
        "the copy-on-write DELETE removed the whole data file, so its DV must be dropped"
    );
    assert_eq!(summary_value(&table, "removed-dvs").as_deref(), Some("1"));
    assert_eq!(
        summary_value(&table, "removed-delete-files").as_deref(),
        Some("1")
    );
    assert_eq!(
        summary_value(&table, "removed-position-deletes").as_deref(),
        Some("1")
    );
    assert_eq!(
        summary_value(&table, "total-delete-files").as_deref(),
        Some("0")
    );
    assert_eq!(live_ids(&fixture.ctx).await, vec![4, 5, 6]);
}
