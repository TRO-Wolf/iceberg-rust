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

use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{DataFile, FormatVersion, ManifestContentType};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;

const EXPECTED_FIXTURES: usize = 1;

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
            Some((cur, _)) if version <= *cur => {}
            _ => best = Some((version, path)),
        }
    }
    best.map(|(_, p)| p)
        .expect("Java table writes vN.metadata.json")
}

async fn live_delete_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut files = Vec::new();
    for mf in manifest_list.entries() {
        if mf.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = mf
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

#[tokio::test]
async fn test_f21_rust_delete_merges_legacy_parquet() {
    let Some(java_dir) = std::env::var_os("ICEBERG_INTEROP_F21_JAVA_SHARED")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
    else {
        println!("skipping F-21 legacy-delete-merge GEN — set ICEBERG_INTEROP_F21_JAVA_SHARED");
        return;
    };
    let metadata_location = current_hadoop_metadata(&java_dir.join("table").join("metadata"));
    assert!(
        metadata_location.is_file(),
        "missing Java Hadoop metadata at {}",
        metadata_location.display()
    );
    let out_dir = java_dir.join("after_delete");
    fs::create_dir_all(out_dir.join("rust_table").join("metadata")).expect("after_delete dir");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_f21_java_shared",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                java_dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("catalog");
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let ident = TableIdent::new(namespace.clone(), "rust_table".to_string());
    catalog
        .register_table(&ident, metadata_location.to_string_lossy().to_string())
        .await
        .expect("register Java table");
    let mut table = catalog.load_table(&ident).await.expect("load Java table");
    let tx = Transaction::new(&table);
    let tx = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3)
        .apply(tx)
        .unwrap();
    table = tx.commit(&catalog).await.unwrap();
    assert_eq!(table.metadata().format_version(), FormatVersion::V3);
    let tx = Transaction::new(&table);
    let tx = tx
        .update_table_properties()
        .set("write.delete.mode".to_string(), "merge-on-read".to_string())
        .set("write.update.mode".to_string(), "merge-on-read".to_string())
        .apply(tx)
        .unwrap();
    tx.commit(&catalog).await.unwrap();
    let client = Arc::new(catalog);
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .expect("provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    ctx.sql("DELETE FROM catalog.interop.rust_table WHERE id = 3")
        .await
        .expect("plan DELETE")
        .collect()
        .await
        .expect("execute DELETE");
    let table = client.load_table(&ident).await.expect("load after delete");
    let deletes = live_delete_files(&table).await;
    assert_eq!(deletes.len(), 1);
    assert_eq!(deletes[0].record_count(), 2);
    fs::write(
        out_dir.join("expected_rows.json"),
        "[\n  {\"id\": 1, \"data\": \"a\"},\n  {\"id\": 4, \"data\": \"d\"}\n]\n",
    )
    .expect("expected_rows.json");
    let final_metadata_path = out_dir
        .join("rust_table")
        .join("metadata")
        .join("final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), final_metadata_path.to_str().expect("utf8"))
        .await
        .expect("write final.metadata.json");
    let fixtures = fs::read_dir(&out_dir)
        .expect("fixture dir")
        .filter(|e| e.as_ref().map(|e| e.path().is_file()).unwrap_or(false))
        .count();
    assert_eq!(
        fixtures,
        EXPECTED_FIXTURES,
        "expected exactly {EXPECTED_FIXTURES} fixture files in {}",
        out_dir.display()
    );
    println!(
        "interop_f21 legacy-delete-merge GEN OK ({fixtures} fixtures) → {}",
        final_metadata_path.display()
    );
}
