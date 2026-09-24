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

use super::catalog::tests::{create_table_with_namespace, new_memory_catalog, temp_path};
use super::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use crate::{
    Catalog, CatalogBuilder, NamespaceIdent, TableCommit, TableIdent, TableMetadataCache,
    TableUpdate,
};

async fn new_memory_catalog_with_cache(cache: Arc<TableMetadataCache>) -> MemoryCatalog {
    let warehouse_location = temp_path();
    MemoryCatalogBuilder::default()
        .with_table_metadata_cache(cache)
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_location)]),
        )
        .await
        .expect("build memory catalog with table metadata cache")
}

#[tokio::test]
async fn test_fk4_1_two_loads_unchanged_pointer_zero_body_fetch() {
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = new_memory_catalog_with_cache(cache.clone()).await;
    let table = create_table_with_namespace(&catalog).await;
    let ident = table.identifier().clone();
    let pointer = table.metadata_location().expect("pointer").to_string();

    cache.reset_stats();
    let first = catalog.load_table(&ident).await.expect("load 1");
    let second = catalog.load_table(&ident).await.expect("load 2");

    let stats = cache.stats();
    assert_eq!(
        stats.body_fetches, 0,
        "unchanged pointer after create-seed must not body-GET on load"
    );
    assert_eq!(stats.hits, 2, "both loads must hit the pointer cache");
    assert_eq!(stats.misses, 0);
    assert_eq!(
        first.metadata_location().unwrap(),
        pointer.as_str(),
        "load must surface the same catalog pointer"
    );
    assert_eq!(second.metadata_location().unwrap(), pointer.as_str());
    assert!(
        std::sync::Arc::ptr_eq(&first.metadata_ref(), &second.metadata_ref()),
        "two loads must share the cached TableMetadata Arc"
    );
}

#[tokio::test]
async fn test_fk4_1_default_off_loads_without_cache() {
    let catalog = new_memory_catalog().await;
    let table = create_table_with_namespace(&catalog).await;
    let loaded = catalog
        .load_table(table.identifier())
        .await
        .expect("load without cache");
    assert_eq!(
        loaded.metadata_location(),
        table.metadata_location(),
        "default-OFF path must still resolve the catalog pointer"
    );
}

#[tokio::test]
async fn test_fk4_1_pointer_change_on_update_is_new_key() {
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = new_memory_catalog_with_cache(cache.clone()).await;
    let table = create_table_with_namespace(&catalog).await;
    let ident = table.identifier().clone();
    let base_location = table.metadata_location().unwrap().to_string();

    let commit = TableCommit::builder()
        .ident(ident.clone())
        .requirements(vec![])
        .updates(vec![TableUpdate::SetProperties {
            updates: HashMap::from([("fk4".to_string(), "1".to_string())]),
        }])
        .base_metadata_location(Some(base_location.clone()))
        .build();
    let updated = catalog.update_table(commit).await.expect("update");
    let new_location = updated.metadata_location().unwrap().to_string();
    assert_ne!(
        new_location, base_location,
        "commit must publish a new metadata pointer"
    );

    cache.reset_stats();
    let loaded = catalog.load_table(&ident).await.expect("load after update");
    assert_eq!(loaded.metadata_location().unwrap(), new_location.as_str());
    assert_eq!(
        loaded
            .metadata()
            .properties()
            .get("fk4")
            .map(String::as_str),
        Some("1")
    );
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.stats().body_fetches, 0);
    assert_eq!(cache.stats().misses, 0);
}

#[tokio::test]
async fn test_fk4_1_invalidate_table_evicts_pointer_entry() {
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = new_memory_catalog_with_cache(cache.clone()).await;
    let table = create_table_with_namespace(&catalog).await;
    let ident = table.identifier().clone();
    let pointer = table.metadata_location().unwrap().to_string();

    cache.reset_stats();
    let _ = catalog.load_table(&ident).await.expect("warm");
    assert_eq!(cache.stats().hits, 1);

    catalog.invalidate_table(&ident).await.expect("invalidate");
    assert!(
        cache
            .lookup(&catalog.cache_scope, &pointer, None)
            .await
            .is_none(),
        "invalidate_table must drop the location entry"
    );

    cache.reset_stats();
    let _ = catalog
        .load_table(&ident)
        .await
        .expect("reload after invalidate");
    assert_eq!(
        cache.stats().body_fetches,
        1,
        "load after invalidate must body-GET (fail closed)"
    );
    assert_eq!(cache.stats().misses, 1);
}

#[tokio::test]
async fn test_fk4_1_reload_same_pointer_is_cache_hit_commit_retry_leg() {
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = new_memory_catalog_with_cache(cache.clone()).await;
    let table = create_table_with_namespace(&catalog).await;
    let ident = table.identifier().clone();

    cache.reset_stats();
    let a = catalog.load_table(&ident).await.expect("retry load 1");
    let b = catalog.load_table(&ident).await.expect("retry load 2");
    assert_eq!(a.metadata_location(), b.metadata_location());
    assert_eq!(cache.stats().hits, 2);
    assert_eq!(
        cache.stats().body_fetches,
        0,
        "commit-retry refresh of unchanged pointer must not re-GET body"
    );
}

#[tokio::test]
async fn test_fk4_1_drop_table_evicts_cache_entry() {
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = new_memory_catalog_with_cache(cache.clone()).await;
    let table = create_table_with_namespace(&catalog).await;
    let ident = table.identifier().clone();
    let pointer = table.metadata_location().unwrap().to_string();
    assert!(
        cache
            .lookup(&catalog.cache_scope, &pointer, None)
            .await
            .is_some()
    );

    catalog.drop_table(&ident).await.expect("drop");
    assert!(
        cache
            .lookup(&catalog.cache_scope, &pointer, None)
            .await
            .is_none(),
        "drop_table must invalidate the metadata-location cache entry"
    );
}

#[tokio::test]
async fn test_fk4_1_invalidate_missing_table_does_not_clear_session() {
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = new_memory_catalog_with_cache(cache.clone()).await;
    let table = create_table_with_namespace(&catalog).await;
    let pointer = table.metadata_location().unwrap().to_string();
    assert!(
        cache
            .lookup(&catalog.cache_scope, &pointer, None)
            .await
            .is_some(),
        "create must seed the cache"
    );

    let missing = TableIdent::new(NamespaceIdent::new("nope".into()), "ghost".into());
    catalog
        .invalidate_table(&missing)
        .await
        .expect("missing invalidate is Ok");
    assert!(
        cache
            .lookup(&catalog.cache_scope, &pointer, None)
            .await
            .is_some(),
        "invalidate of missing table must not clear sibling pointer entries"
    );
}

#[tokio::test]
async fn test_fk4_1_update_evicts_prior_pointer() {
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = new_memory_catalog_with_cache(cache.clone()).await;
    let table = create_table_with_namespace(&catalog).await;
    let ident = table.identifier().clone();
    let base = table.metadata_location().unwrap().to_string();

    let commit = TableCommit::builder()
        .ident(ident.clone())
        .requirements(vec![])
        .updates(vec![TableUpdate::SetProperties {
            updates: HashMap::from([("c6".to_string(), "1".to_string())]),
        }])
        .base_metadata_location(Some(base.clone()))
        .build();
    let updated = catalog.update_table(commit).await.expect("update");
    let new_loc = updated.metadata_location().unwrap().to_string();
    assert_ne!(base, new_loc);
    assert!(
        cache
            .lookup(&catalog.cache_scope, &base, None)
            .await
            .is_none(),
        "prior pointer must be evicted after successful update"
    );
    assert!(
        cache
            .lookup(&catalog.cache_scope, &new_loc, None)
            .await
            .is_some(),
        "new pointer must be seeded"
    );
}
