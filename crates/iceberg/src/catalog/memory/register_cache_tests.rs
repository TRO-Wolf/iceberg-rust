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

use tempfile::TempDir;

use super::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use crate::spec::{NestedField, PrimitiveType, Schema, TableMetadata, TableMetadataBuilder, Type};
use crate::{
    Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent, TableMetadataCache,
};

fn temp_path() -> String {
    let temp_dir = TempDir::new().expect("tempdir");
    temp_dir.keep().to_str().expect("utf8").to_string()
}

fn sample_metadata(location: &str) -> TableMetadata {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("schema");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .location(location.to_string())
        .schema(schema)
        .build();
    TableMetadataBuilder::from_table_creation(creation)
        .expect("builder")
        .build()
        .expect("metadata")
        .metadata
}

fn ident(name: &str) -> TableIdent {
    TableIdent::new(NamespaceIdent::new("ns".to_string()), name.to_string())
}

async fn catalog_with_cache(warehouse: &str, cache: Arc<TableMetadataCache>) -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_table_metadata_cache(cache)
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("build memory catalog")
}

#[tokio::test]
async fn l2_register_reads_body_directly_and_republishes_entry() {
    let warehouse = temp_path();
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = catalog_with_cache(&warehouse, Arc::clone(&cache)).await;
    catalog
        .create_namespace(&NamespaceIdent::new("ns".to_string()), HashMap::new())
        .await
        .expect("create ns");

    let loc = format!("{warehouse}/t/metadata/v1.metadata.json");
    sample_metadata("memory://wh/t-stale")
        .write_to(&catalog.file_io, &loc)
        .await
        .expect("write v1");
    catalog
        .cache_put(&loc, Arc::new(sample_metadata("memory://wh/t-stale")), None)
        .await;

    sample_metadata("memory://wh/t-registered")
        .write_to(&catalog.file_io, &loc)
        .await
        .expect("rewrite body");
    let registered = catalog
        .register_table(&ident("t"), loc.clone())
        .await
        .expect("register");
    assert_eq!(
        registered.metadata().location(),
        "memory://wh/t-registered",
        "register must read the live body, not a cached parse"
    );

    let loaded = catalog
        .load_table(&ident("t"))
        .await
        .expect("load after register");
    assert_eq!(
        loaded.metadata().location(),
        "memory://wh/t-registered",
        "register must republish the entry it replaced"
    );
}

#[tokio::test]
async fn l2_register_missing_body_errors_and_installs_nothing() {
    let warehouse = temp_path();
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = catalog_with_cache(&warehouse, Arc::clone(&cache)).await;
    catalog
        .create_namespace(&NamespaceIdent::new("ns".to_string()), HashMap::new())
        .await
        .expect("create ns");

    let err = catalog
        .register_table(&ident("t"), format!("{warehouse}/t/absent.metadata.json"))
        .await;
    assert!(err.is_err(), "register of a missing body must fail");
    assert_eq!(cache.len(), 0, "a failed read must not install an entry");
}

#[tokio::test]
async fn l2_register_insert_failure_evicts_stale_entry() {
    let warehouse = temp_path();
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = catalog_with_cache(&warehouse, Arc::clone(&cache)).await;

    let loc = format!("{warehouse}/t/metadata/v1.metadata.json");
    sample_metadata("memory://wh/t")
        .write_to(&catalog.file_io, &loc)
        .await
        .expect("write body");
    catalog
        .cache_put(&loc, Arc::new(sample_metadata("memory://wh/t")), None)
        .await;
    cache.run_pending_tasks().await;
    assert_eq!(cache.len(), 1);

    let err = catalog.register_table(&ident("t"), loc).await;
    assert!(err.is_err(), "missing namespace must fail the insert");
    cache.run_pending_tasks().await;
    assert_eq!(
        cache.len(),
        0,
        "a failed register must evict the stale entry"
    );
}
