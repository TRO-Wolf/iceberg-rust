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

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::array::{Array, AsArray};
use datafusion::arrow::datatypes::Int64Type;
use datafusion::execution::context::SessionContext;
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::metadata_columns::RESERVED_COL_NAME_ROW_ID;
use iceberg::spec::{DataContentType, DataFileFormat, FormatVersion};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedRow {
    id: i64,
    val: String,
    row_id: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UpgradeExpectation {
    format_version: u8,
    next_row_id: i64,
    snapshot_sequence_numbers: Vec<i64>,
    rows: Vec<ExpectedRow>,
}

fn upgrade_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_V3_UPGRADE_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

async fn run_sql(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

async fn expectation(table: &Table) -> UpgradeExpectation {
    let batches: Vec<_> = table
        .scan()
        .select(["id", "val", RESERVED_COL_NAME_ROW_ID])
        .build()
        .expect("build the lineage scan")
        .to_arrow()
        .await
        .expect("lineage scan to arrow")
        .try_collect()
        .await
        .expect("collect the lineage batches");
    let mut rows: BTreeMap<i64, ExpectedRow> = BTreeMap::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let vals = batch.column_by_name("val").expect("val column");
        let vals = vals.as_string::<i32>();
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id column")
            .as_primitive::<Int64Type>();
        for index in 0..batch.num_rows() {
            let row_id = if row_ids.is_valid(index) {
                Some(row_ids.value(index))
            } else {
                None
            };
            rows.insert(ids.value(index), ExpectedRow {
                id: ids.value(index),
                val: vals.value(index).to_string(),
                row_id,
            });
        }
    }
    let mut sequence_numbers: Vec<i64> = table
        .metadata()
        .snapshots()
        .map(|snapshot| snapshot.sequence_number())
        .collect();
    sequence_numbers.sort_unstable();
    UpgradeExpectation {
        format_version: match table.metadata().format_version() {
            FormatVersion::V1 => 1,
            FormatVersion::V2 => 2,
            FormatVersion::V3 => 3,
        },
        next_row_id: i64::try_from(table.metadata().next_row_id()).expect("next_row_id fits i64"),
        snapshot_sequence_numbers: sequence_numbers,
        rows: rows.into_values().collect(),
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

fn expectation_json(expectation: &UpgradeExpectation) -> String {
    let sequence_numbers: Vec<String> = expectation
        .snapshot_sequence_numbers
        .iter()
        .map(|value| value.to_string())
        .collect();
    let rows: Vec<String> = expectation
        .rows
        .iter()
        .map(|row| {
            let row_id = match row.row_id {
                Some(value) => value.to_string(),
                None => "null".to_string(),
            };
            format!(
                "{{\"id\":{},\"val\":{},\"row_id\":{}}}",
                row.id,
                json_string(&row.val),
                row_id
            )
        })
        .collect();
    format!(
        "{{\"format_version\":{},\"next_row_id\":{},\"snapshot_sequence_numbers\":[{}],\"rows\":[{}]}}",
        expectation.format_version,
        expectation.next_row_id,
        sequence_numbers.join(","),
        rows.join(",")
    )
}

async fn live_delete_formats(table: &Table) -> Vec<DataFileFormat> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("the table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load the manifest list");
    let mut formats = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load the manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() != DataContentType::Data {
                formats.push(entry.data_file().file_format());
            }
        }
    }
    formats
}

async fn write_final(table: &Table, out_dir: &Path) {
    let dir = out_dir.join("metadata");
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let path = dir.join("final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), path.to_string_lossy().to_string())
        .await
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

#[tokio::test]
async fn gen_rust_merge_on_read_update_over_the_converted_v3_table() {
    let Some(dir) = upgrade_dir() else {
        return;
    };
    let meta = dir
        .join("u3")
        .join("rust_v3_dv")
        .join("metadata")
        .join("final.metadata.json");
    assert!(
        meta.is_file(),
        "missing {} — the core conversion step must run first",
        meta.display()
    );
    let staged = meta
        .parent()
        .expect("metadata dir")
        .join(format!("99999-{}.metadata.json", uuid::Uuid::now_v7()));
    fs::copy(&meta, &staged).unwrap_or_else(|e| panic!("copy {}: {e}", meta.display()));

    let warehouse = dir.to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_v3_upgrade_mor",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse)]),
        )
        .await
        .expect("build the local-fs memory catalog");
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create the namespace");
    let ident = TableIdent::new(namespace, "u3".to_string());
    catalog
        .register_table(&ident, staged.to_string_lossy().to_string())
        .await
        .expect("register the converted V3 table");
    let table = catalog.load_table(&ident).await.expect("load u3");
    assert_eq!(table.metadata().format_version(), FormatVersion::V3);
    let before = expectation(&table).await;
    assert_eq!(
        before.rows.iter().map(|row| row.id).collect::<Vec<_>>(),
        vec![1, 3, 4, 5]
    );

    let tx = Transaction::new(&table);
    tx.update_table_properties()
        .set("write.delete.mode".to_string(), "merge-on-read".to_string())
        .set("write.update.mode".to_string(), "merge-on-read".to_string())
        .apply(tx)
        .expect("apply the merge-on-read properties")
        .commit(&catalog)
        .await
        .expect("commit the merge-on-read properties");

    let client = Arc::new(catalog);
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .expect("build the DataFusion catalog provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    run_sql(&ctx, "UPDATE catalog.interop.u3 SET val = 'X' WHERE id = 3").await;

    let table = client
        .load_table(&ident)
        .await
        .expect("reload after UPDATE");
    let after = expectation(&table).await;
    assert_eq!(
        after.rows.iter().map(|row| row.id).collect::<Vec<_>>(),
        vec![1, 3, 4, 5],
        "the merge-on-read UPDATE replaces the row without changing the live id set"
    );
    assert_eq!(
        after
            .rows
            .iter()
            .find(|row| row.id == 3)
            .and_then(|row| row.row_id),
        before
            .rows
            .iter()
            .find(|row| row.id == 3)
            .and_then(|row| row.row_id),
        "the replacement row keeps the original row id"
    );
    assert_eq!(
        after
            .rows
            .iter()
            .find(|row| row.id == 3)
            .map(|row| row.val.as_str()),
        Some("X")
    );
    let formats = live_delete_formats(&table).await;
    assert!(!formats.is_empty());
    assert!(
        formats
            .iter()
            .all(|format| *format == DataFileFormat::Puffin),
        "a V3 merge-on-read UPDATE must not add a parquet position delete, got {formats:?}"
    );

    write_final(&table, &dir.join("u3").join("rust_v3_mor")).await;
    let expected_path = dir.join("u3").join("rust_expected.json");
    fs::write(&expected_path, expectation_json(&after))
        .unwrap_or_else(|e| panic!("write {}: {e}", expected_path.display()));
}
