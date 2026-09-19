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

use iceberg::io::FileIO;
use iceberg::spec::{NestedField, PrimitiveType, Schema, TableMetadata, TableMetadataBuilder, Type};
use iceberg::{TableCreation, TableMetadataCache};

use super::*;

type PointerFn = Arc<dyn Fn(&TableIdent) -> Result<(String, String)> + Send + Sync>;

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

fn config(arn: &str) -> S3TablesCatalogConfig {
    S3TablesCatalogConfig {
        name: Some("s3t".to_string()),
        table_bucket_arn: arn.to_string(),
        endpoint_url: None,
        client: None,
        props: HashMap::new(),
    }
}

fn ident(name: &str) -> TableIdent {
    TableIdent::new(NamespaceIdent::new("ns".to_string()), name.to_string())
}

fn mutable_pointer(location: &str, token: &str) -> (Arc<Mutex<(String, String)>>, PointerFn) {
    let state = Arc::new(Mutex::new((location.to_string(), token.to_string())));
    let held = Arc::clone(&state);
    (
        state,
        Arc::new(move |_| Ok(held.lock().expect("pointer state").clone())),
    )
}

fn mapped_pointer(entries: &[(&str, &str, &str)]) -> PointerFn {
    let map: HashMap<String, (String, String)> = entries
        .iter()
        .map(|(name, location, token)| {
            (name.to_string(), (location.to_string(), token.to_string()))
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
    arn: &str,
    file_io: &FileIO,
    cache: Option<Arc<TableMetadataCache>>,
    object_cache_bytes: Option<u64>,
    cred_ctx: Option<String>,
    pointer: PointerFn,
) -> S3TablesCatalog {
    S3TablesCatalog::new(config(arn), None)
        .await
        .expect("build catalog")
        .with_cache_options(cache, object_cache_bytes, cred_ctx)
        .with_file_io_for_tests(file_io.clone())
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

    let arn = "arn:aws:s3tables:us-east-1:1:bucket/shared";
    let (state, source) = mutable_pointer(loc1, "tok-1");
    let cache = Arc::new(TableMetadataCache::new());
    let first_handle = catalog(arn, &file_io, Some(Arc::clone(&cache)), None, None, Arc::clone(&source)).await;
    let second_handle = catalog(arn, &file_io, Some(cache), None, None, source).await;
    let t = ident("t");

    let before = first_handle.load_table(&t).await.expect("first load");
    assert_eq!(before.metadata().location(), "memory://wh/t");

    *state.lock().expect("pointer state") = (loc2.to_string(), "tok-2".to_string());
    second_handle.cache_put(loc2, &meta2).await;

    let after = first_handle.load_table(&t).await.expect("load after commit");
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

    let (state, source) = mutable_pointer(loc1, "tok-1");
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/ext",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        source,
    )
    .await;
    let t = ident("t");

    cat.load_table(&t).await.expect("first load");
    *state.lock().expect("pointer state") = (loc2.to_string(), "tok-9".to_string());

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
    let (_state, source) = mutable_pointer(loc, "tok");
    let cache = Arc::new(TableMetadataCache::new());
    let t = ident("t");

    let cat_a = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/a",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        Arc::clone(&source),
    )
    .await;
    let cat_b = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/b",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        Arc::clone(&source),
    )
    .await;
    let cat_a_other_creds = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/a",
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
    let (_state, source) = mutable_pointer(loc, "tok");
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/warm",
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
    let source = mapped_pointer(&[
        ("t1", loc1, "tok1"),
        ("t2", loc2, "tok2"),
    ]);
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/oc",
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
        "arn:aws:s3tables:us-east-1:1:bucket/oc",
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
    let (state, source) = mutable_pointer(loc, "tok-1");
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/vt",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        None,
        source,
    )
    .await;
    let t = ident("t");

    cat.load_table(&t).await.expect("seed v1");
    *state.lock().expect("pointer state") = (loc.to_string(), "tok-2".to_string());
    cat.load_table(&t).await.expect("load with new token");

    let stats = cache.stats();
    assert_eq!(stats.misses, 2, "new version token must fail closed");
    assert_eq!(stats.body_fetches, 2);
    assert_eq!(stats.hits, 0);

    cat.load_table(&t).await.expect("warm same-token load");
    assert_eq!(cache.stats().hits, 1, "stable token hits again");
}

#[tokio::test]
async fn p9_no_cache_handles_every_load_body_gets() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write v1");

    let (_state, source) = mutable_pointer(loc, "tok");
    let uncached = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/raw",
        &file_io,
        None,
        None,
        None,
        Arc::clone(&source),
    )
    .await;
    let cached = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/raw2",
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
