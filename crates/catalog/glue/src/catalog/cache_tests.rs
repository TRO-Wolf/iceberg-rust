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
use std::sync::{Arc, Mutex};

use iceberg::io::{FileIO, MemoryStorageFactory};
use iceberg::spec::{
    NestedField, PrimitiveType, Schema, TableMetadata, TableMetadataBuilder, Type,
};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{CatalogBuilder, TableCreation, TableMetadataCache};

use super::*;
use crate::commit_transport::{GlueCommitScript, ScriptedGlueCommitTransport};

type PointerFn = Arc<dyn Fn(&TableIdent) -> Result<(String, Option<String>)> + Send + Sync>;
type PointerState = Arc<Mutex<(String, Option<String>)>>;

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

fn config(catalog_id: Option<&str>, warehouse: &str) -> GlueCatalogConfig {
    GlueCatalogConfig {
        name: Some("glue".to_string()),
        uri: None,
        catalog_id: catalog_id.map(str::to_string),
        warehouse: warehouse.to_string(),
        props: HashMap::new(),
    }
}

fn ident(name: &str) -> TableIdent {
    TableIdent::new(NamespaceIdent::new("ns".to_string()), name.to_string())
}

fn mutable_pointer(location: &str, version: Option<&str>) -> (PointerState, PointerFn) {
    let state = Arc::new(Mutex::new((
        location.to_string(),
        version.map(str::to_string),
    )));
    let held = Arc::clone(&state);
    (
        state,
        Arc::new(move |_| Ok(held.lock().expect("pointer state").clone())),
    )
}

fn mapped_pointer(entries: &[(&str, &str, &str)]) -> PointerFn {
    let map: HashMap<String, (String, Option<String>)> = entries
        .iter()
        .map(|(name, location, version)| {
            (
                name.to_string(),
                (location.to_string(), Some(version.to_string())),
            )
        })
        .collect();
    Arc::new(move |ident| {
        map.get(ident.name()).cloned().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("no table named {}", ident.name()),
            )
        })
    })
}

async fn catalog(
    catalog_id: Option<&str>,
    warehouse: &str,
    file_io: &FileIO,
    cache: Option<Arc<TableMetadataCache>>,
    object_cache_bytes: Option<u64>,
    cred_ctx: Option<String>,
    pointer: PointerFn,
) -> GlueCatalog {
    GlueCatalog::new(config(catalog_id, warehouse), None)
        .await
        .expect("build catalog")
        .with_file_io_for_tests(file_io.clone())
        .with_cache_options(cache, object_cache_bytes, cred_ctx)
        .with_pointer_source(pointer)
}

#[tokio::test]
async fn p1_second_handle_commit_visible_on_first_handle_next_load() {
    let file_io = FileIO::new_with_memory();
    let loc1 = "memory://wh/t/metadata/v1.metadata.json";
    let loc2 = "memory://wh/t/metadata/v2.metadata.json";
    let meta2 = sample_metadata("memory://wh/t-after-commit");
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc1)
        .await
        .expect("write v1");
    meta2.write_to(&file_io, loc2).await.expect("write v2");

    let (state, source) = mutable_pointer(loc1, Some("vid-1"));
    let cache = Arc::new(TableMetadataCache::new());
    let first_handle = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        Arc::clone(&source),
    )
    .await;
    let second_handle = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(cache),
        None,
        None,
        source,
    )
    .await;
    let t = ident("t");

    let before = first_handle.load_table(&t).await.expect("first load");
    assert_eq!(before.metadata().location(), "memory://wh/t");

    *state.lock().expect("pointer state") = (loc2.to_string(), Some("vid-2".to_string()));
    second_handle
        .cache_put(loc2, Arc::new(meta2), Some("vid-2".to_string()))
        .await;

    let after = first_handle
        .load_table(&t)
        .await
        .expect("load after commit");
    assert_eq!(
        after.metadata_location(),
        Some(loc2),
        "first handle must follow the moved service pointer"
    );
    assert_eq!(after.metadata().location(), "memory://wh/t-after-commit");
}

#[tokio::test]
async fn p2_external_pointer_move_visible_on_next_load() {
    let file_io = FileIO::new_with_memory();
    let loc1 = "memory://wh/t/metadata/v1.metadata.json";
    let loc2 = "memory://wh/t/metadata/v9.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc1)
        .await
        .expect("write v1");
    sample_metadata("memory://wh/t-external")
        .write_to(&file_io, loc2)
        .await
        .expect("write v9");

    let (state, source) = mutable_pointer(loc1, Some("vid-1"));
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        source,
    )
    .await;
    let t = ident("t");

    cat.load_table(&t).await.expect("first load");
    *state.lock().expect("pointer state") = (loc2.to_string(), Some("vid-9".to_string()));

    let loaded = cat.load_table(&t).await.expect("load after external move");
    assert_eq!(loaded.metadata_location(), Some(loc2));
    assert_eq!(loaded.metadata().location(), "memory://wh/t-external");
    let stats = cache.stats();
    assert_eq!(stats.misses, 2, "moved pointer must miss the cache");
    assert_eq!(stats.body_fetches, 2);
}

#[tokio::test]
async fn p4_same_location_under_different_scopes_never_shared() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write");
    let (_state, source) = mutable_pointer(loc, Some("vid"));
    let cache = Arc::new(TableMetadataCache::new());
    let t = ident("t");

    let cat_a = catalog(
        Some("cat-a"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        Arc::clone(&source),
    )
    .await;
    let cat_b = catalog(
        Some("cat-b"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        Arc::clone(&source),
    )
    .await;
    let cat_a_other_creds = catalog(
        Some("cat-a"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        Some("ctx-other".to_string()),
        source,
    )
    .await;

    cat_a.load_table(&t).await.expect("load a");
    cat_b.load_table(&t).await.expect("load b");
    cat_a_other_creds.load_table(&t).await.expect("load a2");

    let stats = cache.stats();
    assert_eq!(stats.misses, 3, "each scope must fetch its own entry");
    assert_eq!(stats.body_fetches, 3);
    assert_eq!(stats.hits, 0);
    cache.run_pending_tasks().await;
    assert_eq!(cache.len(), 3, "one entry per (scope, location)");
}

#[tokio::test]
async fn p6_warm_reload_zero_body_gets_same_arc() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write");
    let (_state, source) = mutable_pointer(loc, Some("vid"));
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        source,
    )
    .await;
    let t = ident("t");

    let first = cat.load_table(&t).await.expect("cold load");
    cache.reset_stats();
    let second = cat.load_table(&t).await.expect("warm load");

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.body_fetches, 0, "warm reload must not body-GET");
    assert!(
        Arc::ptr_eq(&first.metadata_ref(), &second.metadata_ref()),
        "warm load must return the cached Arc"
    );
}

#[tokio::test]
async fn p7_object_cache_shared_within_catalog_not_across_instances() {
    let file_io = FileIO::new_with_memory();
    let loc1 = "memory://wh/t1/metadata/v1.metadata.json";
    let loc2 = "memory://wh/t2/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t1")
        .write_to(&file_io, loc1)
        .await
        .expect("w1");
    sample_metadata("memory://wh/t2")
        .write_to(&file_io, loc2)
        .await
        .expect("w2");
    let source = mapped_pointer(&[("t1", loc1, "v1"), ("t2", loc2, "v2")]);
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(cache),
        Some(1 << 20),
        None,
        Arc::clone(&source),
    )
    .await;

    let t1 = cat.load_table(&ident("t1")).await.expect("load t1");
    let t2 = cat.load_table(&ident("t2")).await.expect("load t2");
    assert!(
        Arc::ptr_eq(&t1.object_cache(), &t2.object_cache()),
        "every table from one catalog must share its ObjectCache"
    );
    assert!(
        Arc::ptr_eq(
            &t1.object_cache(),
            cat.shared_object_cache.as_ref().expect("object cache")
        ),
        "table cache must be the catalog instance cache"
    );

    let other = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        None,
        Some(1 << 20),
        None,
        source,
    )
    .await;
    let t3 = other.load_table(&ident("t1")).await.expect("load other");
    assert!(
        !Arc::ptr_eq(&t1.object_cache(), &t3.object_cache()),
        "two catalog instances must never share an ObjectCache"
    );
}

#[tokio::test]
async fn p8_version_token_change_forces_refetch() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write");
    let (state, source) = mutable_pointer(loc, Some("vid-1"));
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        source,
    )
    .await;
    let t = ident("t");

    cat.load_table(&t).await.expect("seed v1");
    *state.lock().expect("pointer state") = (loc.to_string(), Some("vid-2".to_string()));
    cat.load_table(&t).await.expect("load with new version");

    let stats = cache.stats();
    assert_eq!(stats.misses, 2, "new version id must fail closed");
    assert_eq!(stats.body_fetches, 2);
    assert_eq!(stats.hits, 0);

    cat.load_table(&t).await.expect("warm same-version load");
    assert_eq!(cache.stats().hits, 1, "stable version hits again");
}

#[tokio::test]
async fn p9_no_cache_handles_every_load_body_gets() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write v1");

    let (_state, source) = mutable_pointer(loc, Some("vid"));
    let uncached = catalog(
        Some("cat-raw"),
        "s3://wh",
        &file_io,
        None,
        None,
        None,
        Arc::clone(&source),
    )
    .await;
    let cached = catalog(
        Some("cat-warm"),
        "s3://wh",
        &file_io,
        Some(Arc::new(TableMetadataCache::new())),
        None,
        None,
        source,
    )
    .await;
    let t = ident("t");

    let first = uncached.load_table(&t).await.expect("load 1");
    assert_eq!(first.metadata().location(), "memory://wh/t");
    assert!(uncached.table_metadata_cache.is_none());

    sample_metadata("memory://wh/t-rewritten")
        .write_to(&file_io, loc)
        .await
        .expect("rewrite body");

    let second = uncached.load_table(&t).await.expect("load 2");
    assert_eq!(
        second.metadata().location(),
        "memory://wh/t-rewritten",
        "without a cache handle every load re-reads the body"
    );

    let warm = cached.load_table(&t).await.expect("cached first load");
    assert_eq!(warm.metadata().location(), "memory://wh/t-rewritten");
    sample_metadata("memory://wh/t-rewritten-again")
        .write_to(&file_io, loc)
        .await
        .expect("rewrite body again");
    let warm2 = cached.load_table(&t).await.expect("cached second load");
    assert_eq!(
        warm2.metadata().location(),
        "memory://wh/t-rewritten",
        "with a cache handle the second load serves the cached Arc"
    );
}

fn builder_props(
    catalog_id: Option<&str>,
    warehouse: &str,
    extra: &[(&str, &str)],
) -> HashMap<String, String> {
    let mut props = HashMap::new();
    if let Some(id) = catalog_id {
        props.insert(GLUE_CATALOG_PROP_CATALOG_ID.to_string(), id.to_string());
    }
    props.insert(
        GLUE_CATALOG_PROP_WAREHOUSE.to_string(),
        warehouse.to_string(),
    );
    for (key, value) in extra {
        props.insert(key.to_string(), value.to_string());
    }
    props
}

type FailablePointerState = Arc<Mutex<Option<(String, Option<String>)>>>;

fn failable_pointer() -> (FailablePointerState, PointerFn) {
    let state = Arc::new(Mutex::new(None));
    let held = Arc::clone(&state);
    (
        state,
        Arc::new(move |_| {
            held.lock()
                .expect("pointer state")
                .clone()
                .ok_or_else(|| Error::new(ErrorKind::Unexpected, "pointer fetch failed"))
        }),
    )
}

#[tokio::test]
async fn l1_region_only_injected_factory_isolates_shared_cache() {
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    let (_state, source) = mutable_pointer(loc, Some("vid"));
    let shared = Arc::new(TableMetadataCache::new());
    let t = ident("t");

    let cat_a = GlueCatalogBuilder::default()
        .with_table_metadata_cache(Arc::clone(&shared))
        .with_storage_factory(Arc::new(MemoryStorageFactory))
        .load(
            "glue-a",
            builder_props(Some("shared-cat"), "memory://wh", &[(
                "region_name",
                "us-east-1",
            )]),
        )
        .await
        .expect("load catalog a")
        .with_pointer_source(Arc::clone(&source));
    let cat_b = GlueCatalogBuilder::default()
        .with_table_metadata_cache(Arc::clone(&shared))
        .with_storage_factory(Arc::new(MemoryStorageFactory))
        .load(
            "glue-b",
            builder_props(Some("shared-cat"), "memory://wh", &[(
                "region_name",
                "us-east-1",
            )]),
        )
        .await
        .expect("load catalog b")
        .with_pointer_source(source);

    sample_metadata("memory://wh/t-a")
        .write_to(&cat_a.file_io(), loc)
        .await
        .expect("write a body");
    sample_metadata("memory://wh/t-b")
        .write_to(&cat_b.file_io(), loc)
        .await
        .expect("write b body");

    let table_a = cat_a.load_table(&t).await.expect("load a");
    let table_b = cat_b.load_table(&t).await.expect("load b");
    assert_eq!(table_a.metadata().location(), "memory://wh/t-a");
    assert_eq!(table_b.metadata().location(), "memory://wh/t-b");
    let stats = shared.stats();
    assert_eq!(stats.body_fetches, 2, "injected-io scopes must isolate");
    assert_eq!(stats.misses, 2);

    cat_a.load_table(&t).await.expect("warm a load");
    assert_eq!(shared.stats().hits, 1, "each isolated scope still caches");
}

#[tokio::test]
async fn l2_register_table_reads_body_directly() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    let (_state, source) = mutable_pointer(loc, Some("vid"));
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        source,
    )
    .await
    .with_create_source(Arc::new(|_| Ok(())));
    let t = ident("t");

    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write v1");
    cat.load_table(&t).await.expect("seed cached entry");

    sample_metadata("memory://wh/t-registered")
        .write_to(&file_io, loc)
        .await
        .expect("rewrite body");
    let registered = cat
        .register_table(&ident("t2"), loc.to_string())
        .await
        .expect("register reads the live body");
    assert_eq!(
        registered.metadata().location(),
        "memory://wh/t-registered",
        "register must bypass the cached parse"
    );

    let loaded = cat.load_table(&t).await.expect("load after register");
    assert_eq!(
        loaded.metadata().location(),
        "memory://wh/t-registered",
        "register must republish the entry it replaced"
    );

    let missing = cat
        .register_table(
            &ident("t3"),
            "memory://wh/t/absent.metadata.json".to_string(),
        )
        .await;
    assert!(missing.is_err(), "register of a missing body must fail");
}

#[tokio::test]
async fn l3_invalidate_table_evicts_current_pointer_location() {
    let file_io = FileIO::new_with_memory();
    let loc1 = "memory://wh/t/metadata/v1.metadata.json";
    let loc2 = "memory://wh/t/metadata/v2.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc1)
        .await
        .expect("write v1");
    sample_metadata("memory://wh/t-moved")
        .write_to(&file_io, loc2)
        .await
        .expect("write v2");
    let (state, source) = mutable_pointer(loc1, Some("vid-1"));
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        source,
    )
    .await;
    let t = ident("t");

    cat.load_table(&t).await.expect("seed loc1 entry");
    *state.lock().expect("pointer state") = (loc2.to_string(), Some("vid-2".to_string()));
    cat.load_table(&t).await.expect("seed loc2 entry");

    cat.invalidate_table(&t).await.expect("invalidate");
    cat.load_table(&t).await.expect("load after invalidate");

    let stats = cache.stats();
    assert_eq!(
        stats.misses, 3,
        "the evicted current location must miss again"
    );
    assert_eq!(stats.body_fetches, 3);
    cache.run_pending_tasks().await;
    assert_eq!(
        cache.len(),
        2,
        "loc1 entry stays; only the current location evicts"
    );
}

#[tokio::test]
async fn l3_invalidate_pointer_failure_evicts_nothing() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write");
    let (state, source) = failable_pointer();
    *state.lock().expect("pointer state") = Some((loc.to_string(), Some("vid".to_string())));
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        source,
    )
    .await;
    let t = ident("t");

    cat.load_table(&t).await.expect("seed entry");
    *state.lock().expect("pointer state") = None;

    let err = cat.invalidate_table(&t).await;
    assert!(err.is_err(), "a failed pointer fetch must surface");

    *state.lock().expect("pointer state") = Some((loc.to_string(), Some("vid".to_string())));
    cat.load_table(&t).await.expect("warm load still cached");
    assert_eq!(cache.stats().hits, 1, "nothing must be evicted on failure");
}

#[tokio::test]
async fn l3_drop_table_evicts_last_known_location() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write");
    let (_state, source) = mutable_pointer(loc, Some("vid"));
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        source,
    )
    .await
    .with_drop_source(Arc::new(|_| Ok(())));
    let t = ident("t");

    cat.load_table(&t).await.expect("seed entry");
    cat.drop_table(&t).await.expect("drop");
    cat.load_table(&t).await.expect("load after drop");

    let stats = cache.stats();
    assert_eq!(stats.misses, 2, "dropped location must refetch");
    assert_eq!(stats.body_fetches, 2);
}

#[tokio::test]
async fn l3_drop_failure_keeps_cache_entry() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write");
    let (_state, source) = mutable_pointer(loc, Some("vid"));
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        source,
    )
    .await
    .with_drop_source(Arc::new(|_| {
        Err(Error::new(ErrorKind::Unexpected, "delete refused"))
    }));
    let t = ident("t");

    cat.load_table(&t).await.expect("seed entry");
    let err = cat.drop_table(&t).await;
    assert!(err.is_err(), "failed drop must surface");
    cat.load_table(&t)
        .await
        .expect("warm load after failed drop");
    assert_eq!(cache.stats().hits, 1, "a failed drop must not evict");
}

#[tokio::test]
async fn l005_publish_arms_version_id_on_first_load() {
    let file_io = FileIO::new_with_memory();
    let loc1 = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc1)
        .await
        .expect("write v1");
    let (state, source) = mutable_pointer(loc1, Some("vid-1"));
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        source,
    )
    .await
    .with_commit_transport(ScriptedGlueCommitTransport::new([
        GlueCommitScript::Success,
    ]));
    let t = ident("t");

    let table = cat.load_table(&t).await.expect("seed load");
    let tx = Transaction::new(&table);
    let committed = tx
        .update_table_properties()
        .set("v".to_string(), "2".to_string())
        .apply(tx)
        .expect("apply")
        .commit(&cat)
        .await
        .expect("commit");
    let loc2 = committed
        .metadata_location()
        .expect("committed location")
        .to_string();

    *state.lock().expect("pointer state") = (loc2.clone(), Some("vid-2".to_string()));
    cat.load_table(&t).await.expect("published entry hits");
    let stats = cache.stats();
    assert_eq!(
        stats.body_fetches, 1,
        "the published entry must serve the load"
    );
    assert_eq!(stats.hits, 2, "commit base refresh + published load");

    *state.lock().expect("pointer state") = (loc2.clone(), Some("vid-3".to_string()));
    cat.load_table(&t)
        .await
        .expect("a different version id must fail closed");
    let stats = cache.stats();
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.body_fetches, 2);

    cat.load_table(&t).await.expect("re-armed warm load");
    assert_eq!(cache.stats().hits, 3);
}

#[tokio::test]
async fn p1_commit_through_update_table_visible_on_shared_handle() {
    let file_io = FileIO::new_with_memory();
    let loc1 = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc1)
        .await
        .expect("write v1");
    let (state, source) = mutable_pointer(loc1, Some("vid-1"));
    let cache = Arc::new(TableMetadataCache::new());
    let cat_a = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        Some("ctx-shared".to_string()),
        Arc::clone(&source),
    )
    .await;
    let cat_b = catalog(
        Some("cat-1"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        Some("ctx-shared".to_string()),
        source,
    )
    .await
    .with_commit_transport(ScriptedGlueCommitTransport::new([
        GlueCommitScript::Success,
    ]));
    let t = ident("t");

    cat_a.load_table(&t).await.expect("cold load on a");
    let table_b = cat_b.load_table(&t).await.expect("warm load on b");
    assert_eq!(cache.stats().body_fetches, 1);

    let tx = Transaction::new(&table_b);
    let committed = tx
        .update_table_properties()
        .set("commit.marker".to_string(), "yes".to_string())
        .apply(tx)
        .expect("apply")
        .commit(&cat_b)
        .await
        .expect("commit through update_table");
    let loc2 = committed
        .metadata_location()
        .expect("committed location")
        .to_string();
    assert_ne!(loc2, loc1);

    *state.lock().expect("pointer state") = (loc2.clone(), Some("vid-2".to_string()));

    let after = cat_a.load_table(&t).await.expect("load after commit");
    assert_eq!(after.metadata_location(), Some(loc2.as_str()));
    assert_eq!(
        after.metadata().properties().get("commit.marker"),
        Some(&"yes".to_string()),
        "handle a must observe the committed metadata"
    );
    let stats = cache.stats();
    assert_eq!(
        stats.body_fetches, 1,
        "commit must publish the staged metadata so handle a loads warm"
    );
    assert_eq!(
        stats.hits, 3,
        "warm b load + commit base refresh + warm a post-commit load"
    );
}

#[tokio::test]
async fn p4_identical_credential_props_separate_by_identity_alone() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write");
    let (_state, source) = mutable_pointer(loc, Some("vid"));
    let cache = Arc::new(TableMetadataCache::new());
    let t = ident("t");

    let cat_a = catalog(
        Some("cat-a"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        Some("identical-ctx".to_string()),
        Arc::clone(&source),
    )
    .await;
    let cat_b = catalog(
        Some("cat-b"),
        "s3://wh",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        Some("identical-ctx".to_string()),
        source,
    )
    .await;

    cat_a.load_table(&t).await.expect("load a");
    cat_b.load_table(&t).await.expect("load b");

    let stats = cache.stats();
    assert_eq!(
        stats.misses, 2,
        "identical credentials must still separate by catalog identity"
    );
    assert_eq!(stats.body_fetches, 2);
}

#[tokio::test]
async fn p9_builder_load_without_cache_rereads_every_load() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write v1");
    let (_state, source) = mutable_pointer(loc, Some("vid"));

    let cat = GlueCatalogBuilder::default()
        .load("glue-plain", builder_props(Some("cat-1"), "s3://wh", &[]))
        .await
        .expect("builder load")
        .with_file_io_for_tests(file_io.clone())
        .with_pointer_source(source);
    let t = ident("t");

    let first = cat.load_table(&t).await.expect("first load");
    assert_eq!(first.metadata().location(), "memory://wh/t");
    assert!(cat.table_metadata_cache.is_none());

    sample_metadata("memory://wh/t-rewritten")
        .write_to(&file_io, loc)
        .await
        .expect("rewrite body");
    let second = cat.load_table(&t).await.expect("second load");
    assert_eq!(
        second.metadata().location(),
        "memory://wh/t-rewritten",
        "without a cache handle every load re-reads the body"
    );
}
