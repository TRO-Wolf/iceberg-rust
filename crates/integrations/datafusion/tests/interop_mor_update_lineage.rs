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
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::array::{Array, AsArray};
use datafusion::execution::context::SessionContext;
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_COL_NAME_ROW_ID,
};
use iceberg::table::Table;
use iceberg::transaction::ApplyTransactionAction;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_MOR_UPDATE_LINEAGE_GEN_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn current_hadoop_metadata(meta_dir: &Path) -> PathBuf {
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in fs::read_dir(meta_dir).expect("java metadata dir") {
        let path = entry.expect("dirent").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let Some(rest) = name.strip_prefix('v') else {
            continue;
        };
        let Some(digits) = rest.strip_suffix(".metadata.json") else {
            continue;
        };
        let Ok(version) = digits.parse::<u64>() else {
            continue;
        };
        match &best {
            Some((current, _)) if version <= *current => {}
            _ => best = Some((version, path)),
        }
    }
    best.map(|(_, path)| path)
        .expect("Java table writes vN.metadata.json")
}

async fn run_sql(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

async fn lineage_pairs(table: &Table) -> Vec<(i32, i64, i64)> {
    let batches: Vec<_> = table
        .scan()
        .select([
            "id",
            RESERVED_COL_NAME_ROW_ID,
            RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
        ])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id")
            .as_primitive::<datafusion::arrow::datatypes::Int32Type>();
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id")
            .as_primitive::<datafusion::arrow::datatypes::Int64Type>();
        let seqs = batch
            .column_by_name(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER)
            .expect("seq")
            .as_primitive::<datafusion::arrow::datatypes::Int64Type>();
        for index in 0..batch.num_rows() {
            assert!(row_ids.is_valid(index));
            assert!(seqs.is_valid(index));
            rows.push((ids.value(index), row_ids.value(index), seqs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

async fn write_final(table: &Table, out_dir: &Path) {
    fs::create_dir_all(out_dir.join("rust_table").join("metadata")).expect("out dir");
    let final_metadata_path = out_dir
        .join("rust_table")
        .join("metadata")
        .join("final.metadata.json");
    table
        .metadata()
        .write_to(
            table.file_io(),
            final_metadata_path.to_str().expect("utf8 metadata path"),
        )
        .await
        .expect("write final.metadata.json");
}

#[tokio::test]
async fn test_mor_update_lineage_gen() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_mor_update_lineage GEN — set ICEBERG_INTEROP_MOR_UPDATE_LINEAGE_GEN_DIR"
        );
        return;
    };

    let mor_meta = current_hadoop_metadata(&gen_dir.join("mor_table").join("metadata"));
    let cow_meta = current_hadoop_metadata(&gen_dir.join("cow_table").join("metadata"));
    assert!(mor_meta.is_file(), "missing {}", mor_meta.display());
    assert!(cow_meta.is_file(), "missing {}", cow_meta.display());

    let warehouse = gen_dir.to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_mor_update_lineage",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse)]),
        )
        .await
        .expect("catalog");
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");

    let mor_ident = TableIdent::new(namespace.clone(), "mor_table".to_string());
    catalog
        .register_table(&mor_ident, mor_meta.to_string_lossy().to_string())
        .await
        .expect("register mor");
    let mor_table = catalog.load_table(&mor_ident).await.expect("load mor");
    let tx = iceberg::transaction::Transaction::new(&mor_table);
    tx.update_table_properties()
        .set("write.delete.mode".to_string(), "merge-on-read".to_string())
        .set("write.update.mode".to_string(), "merge-on-read".to_string())
        .apply(tx)
        .expect("apply MoR properties")
        .commit(&catalog)
        .await
        .expect("commit MoR properties");

    let cow_ident = TableIdent::new(namespace.clone(), "cow_table".to_string());
    catalog
        .register_table(&cow_ident, cow_meta.to_string_lossy().to_string())
        .await
        .expect("register cow");

    let client = Arc::new(catalog);
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .expect("provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));

    run_sql(
        &ctx,
        "UPDATE catalog.interop.mor_table SET val = 'B' WHERE id = 2",
    )
    .await;
    run_sql(
        &ctx,
        "UPDATE catalog.interop.mor_table SET val = 'BB' WHERE id = 2",
    )
    .await;

    let mor_table = client.load_table(&mor_ident).await.expect("reload mor");
    let mor_after = gen_dir.join("mor_after");
    write_final(&mor_table, &mor_after).await;
    assert_eq!(mor_table.metadata().next_row_id(), 3);

    run_sql(
        &ctx,
        "UPDATE catalog.interop.cow_table SET val = 'B' WHERE id = 2",
    )
    .await;
    let cow_table = client.load_table(&cow_ident).await.expect("reload cow ow");
    assert_eq!(cow_table.metadata().next_row_id(), 3);
    assert_eq!(lineage_pairs(&cow_table).await, vec![
        (1, 0, 1),
        (2, 1, 2),
        (3, 2, 1)
    ]);
    run_sql(&ctx, "DELETE FROM catalog.interop.cow_table WHERE id = 2").await;
    let cow_table = client.load_table(&cow_ident).await.expect("reload cow del");
    let cow_after = gen_dir.join("cow_after");
    write_final(&cow_table, &cow_after).await;
    assert_eq!(cow_table.metadata().next_row_id(), 5);
    assert_eq!(lineage_pairs(&cow_table).await, vec![(1, 0, 1), (3, 2, 1)]);
}
