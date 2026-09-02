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
use iceberg::spec::{DataFile, ManifestContentType};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;

const EXPECTED_FIXTURES: usize = 4;

fn current_hadoop_metadata(meta_dir: &Path) -> PathBuf {
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in fs::read_dir(meta_dir).expect("java metadata dir") {
        let path = entry.expect("dirent").path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
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
            .expect("delete manifest");
        for entry in manifest.entries() {
            if entry.is_alive() {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

fn dv_entry_json(files: &[DataFile]) -> String {
    let mut rows: Vec<String> = files
        .iter()
        .filter_map(|file| {
            file.referenced_data_file().map(|referenced| {
                format!(
                    "  {{\"referenced\": \"{}\", \"container\": \"{}\", \"offset\": {}, \"size\": {}}}",
                    referenced,
                    file.file_path(),
                    file.content_offset().expect("dv content offset"),
                    file.content_size_in_bytes().expect("dv content size")
                )
            })
        })
        .collect();
    rows.sort();
    format!("[\n{}\n]\n", rows.join(",\n"))
}

fn summary_json(table: &Table) -> String {
    let summary = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .summary();
    let read = |key: &str| -> i64 {
        summary
            .additional_properties
            .get(key)
            .map(|value| value.parse::<i64>().expect("numeric summary value"))
            .unwrap_or(0)
    };
    format!(
        "{{\"removed-delete-files\": {}, \"removed-dvs\": {}, \"added-delete-files\": {}}}\n",
        read("removed-delete-files"),
        read("removed-dvs"),
        read("added-delete-files")
    )
}

#[tokio::test]
async fn test_f18_rust_delete_leaves_the_java_sibling_entry_in_place() {
    let Some(java_dir) = std::env::var_os("ICEBERG_INTEROP_F18_JAVA_SHARED")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    else {
        println!("skipping F-18 sibling-close GEN — set ICEBERG_INTEROP_F18_JAVA_SHARED");
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
            "interop_f18_java_shared",
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
    let table = catalog.load_table(&ident).await.expect("load Java table");

    let before = live_delete_files(&table).await;
    assert_eq!(before.len(), 2, "the Java seed writes two DV blobs");
    let containers: std::collections::BTreeSet<&str> =
        before.iter().map(|file| file.file_path()).collect();
    assert_eq!(containers.len(), 1, "the Java seed writes ONE Puffin");
    fs::write(out_dir.join("before_dvs.json"), dv_entry_json(&before)).expect("before_dvs.json");

    let tx = Transaction::new(&table);
    tx.update_table_properties()
        .set("write.delete.mode".to_string(), "merge-on-read".to_string())
        .set("write.update.mode".to_string(), "merge-on-read".to_string())
        .apply(tx)
        .expect("apply MoR properties")
        .commit(&catalog)
        .await
        .expect("commit MoR properties");

    let client = Arc::new(catalog);
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .expect("provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    ctx.sql("DELETE FROM catalog.interop.rust_table WHERE id = 10")
        .await
        .expect("plan DELETE")
        .collect()
        .await
        .expect("execute DELETE");

    let table = client.load_table(&ident).await.expect("load after delete");
    let after = live_delete_files(&table).await;
    assert_eq!(after.len(), 2, "still one DV per referenced data file");
    fs::write(out_dir.join("after_dvs.json"), dv_entry_json(&after)).expect("after_dvs.json");
    fs::write(out_dir.join("summary.json"), summary_json(&table)).expect("summary.json");
    fs::write(
        out_dir.join("expected_rows.json"),
        "[\n  {\"id\": 30, \"data\": \"z\"},\n  {\"id\": 50, \"data\": \"q\"}\n]\n",
    )
    .expect("expected_rows.json");

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

    let fixtures = fs::read_dir(&out_dir)
        .expect("fixture dir")
        .filter(|entry| {
            entry
                .as_ref()
                .map(|entry| entry.path().is_file())
                .unwrap_or(false)
        })
        .count();
    assert_eq!(
        fixtures,
        EXPECTED_FIXTURES,
        "expected exactly {EXPECTED_FIXTURES} fixture files in {}",
        out_dir.display()
    );
    println!(
        "interop_f18 sibling-close GEN OK ({fixtures} fixtures) → {}",
        final_metadata_path.display()
    );
}
