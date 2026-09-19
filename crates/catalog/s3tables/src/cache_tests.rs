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
use iceberg::spec::{NestedField, PrimitiveType, Schema, TableMetadata, TableMetadataBuilder, Type};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{CatalogBuilder, TableCreation, TableMetadataCache};

use super::*;
use crate::commit_transport::{S3TablesCommitScript, ScriptedS3TablesCommitTransport};
use crate::{S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN, S3TablesCatalogBuilder};

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
    second_handle
        .cache_put(loc2, Arc::new(meta2), Some("tok-2".to_string()))
        .await;

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

fn builder_props(arn: &str, extra: &[(&str, &str)]) -> HashMap<String, String> {
    let mut props = HashMap::from([(
        S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN.to_string(),
        arn.to_string(),
    )]);
    for (key, value) in extra {
        props.insert(key.to_string(), value.to_string());
    }
    props
}

async fn dummy_client() -> aws_sdk_s3tables::Client {
    let cfg = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .credentials_provider(aws_sdk_s3tables::config::Credentials::new(
            "test", "test", None, None, "test",
        ))
        .region(aws_config::Region::new("us-east-1"))
        .load()
        .await;
    aws_sdk_s3tables::Client::new(&cfg)
}

type PointerState = Arc<Mutex<Option<(String, String)>>>;

fn failable_pointer() -> (PointerState, PointerFn) {
    let state = Arc::new(Mutex::new(None));
    let held = Arc::clone(&state);
    (
        state,
        Arc::new(move |_| {
            held.lock().expect("pointer state").clone().ok_or_else(|| {
                Error::new(ErrorKind::Unexpected, "pointer fetch failed")
            })
        }),
    )
}

#[tokio::test]
async fn l1_region_only_injected_io_isolates_shared_cache() {
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    let (_state, source) = mutable_pointer(loc, "tok");
    let shared = Arc::new(TableMetadataCache::new());
    let t = ident("t");

    let cat_factory = S3TablesCatalogBuilder::default()
        .with_table_metadata_cache(Arc::clone(&shared))
        .with_storage_factory(Arc::new(MemoryStorageFactory))
        .load(
            "cat-factory",
            builder_props(
                "arn:aws:s3tables:us-east-1:1:bucket/shared",
                &[("region_name", "us-east-1")],
            ),
        )
        .await
        .expect("load factory catalog")
        .with_pointer_source(Arc::clone(&source));

    let client_io = FileIO::new_with_memory();
    let cat_client = S3TablesCatalogBuilder::default()
        .with_table_metadata_cache(Arc::clone(&shared))
        .with_client(dummy_client().await)
        .load(
            "cat-client",
            builder_props(
                "arn:aws:s3tables:us-east-1:1:bucket/shared",
                &[("region_name", "us-east-1")],
            ),
        )
        .await
        .expect("load client catalog")
        .with_file_io_for_tests(client_io.clone())
        .with_pointer_source(source);

    sample_metadata("memory://wh/t-factory")
        .write_to(&cat_factory.file_io, loc)
        .await
        .expect("write factory body");
    sample_metadata("memory://wh/t-client")
        .write_to(&client_io, loc)
        .await
        .expect("write client body");

    let factory_table = cat_factory.load_table(&t).await.expect("factory load");
    let client_table = cat_client.load_table(&t).await.expect("client load");

    assert_eq!(factory_table.metadata().location(), "memory://wh/t-factory");
    assert_eq!(client_table.metadata().location(), "memory://wh/t-client");
    let stats = shared.stats();
    assert_eq!(stats.body_fetches, 2, "injected-io scopes must isolate");
    assert_eq!(stats.misses, 2);

    cat_factory.load_table(&t).await.expect("warm factory");
    assert_eq!(
        shared.stats().hits,
        1,
        "each isolated scope still caches its own entry"
    );
}

#[tokio::test]
async fn l1_shared_credential_injected_io_still_isolates() {
    for (selector, marker_a, marker_b) in [
        ("aws_access_key_id", "memory://wh/t-akid-factory", "memory://wh/t-akid-client"),
        ("profile_name", "memory://wh/t-prof-factory", "memory://wh/t-prof-client"),
    ] {
        let loc = "memory://wh/t/metadata/v1.metadata.json";
        let (_state, source) = mutable_pointer(loc, "tok");
        let shared = Arc::new(TableMetadataCache::new());
        let t = ident("t");
        let props = builder_props(
            "arn:aws:s3tables:us-east-1:1:bucket/shared",
            &[(selector, "SHARED-CRED")],
        );

        let cat_factory = S3TablesCatalogBuilder::default()
            .with_table_metadata_cache(Arc::clone(&shared))
            .with_storage_factory(Arc::new(MemoryStorageFactory))
            .load("cat-factory", props.clone())
            .await
            .expect("load factory catalog")
            .with_pointer_source(Arc::clone(&source));

        let cat_client = S3TablesCatalogBuilder::default()
            .with_table_metadata_cache(Arc::clone(&shared))
            .with_client(dummy_client().await)
            .with_storage_factory(Arc::new(MemoryStorageFactory))
            .load("cat-client", props)
            .await
            .expect("load client catalog")
            .with_pointer_source(source);

        sample_metadata(marker_a)
            .write_to(&cat_factory.file_io, loc)
            .await
            .expect("write factory body");
        sample_metadata(marker_b)
            .write_to(&cat_client.file_io, loc)
            .await
            .expect("write client body");

        let factory_table = cat_factory.load_table(&t).await.expect("factory load");
        let client_table = cat_client.load_table(&t).await.expect("client load");

        assert_eq!(factory_table.metadata().location(), marker_a, "{selector}");
        assert_eq!(client_table.metadata().location(), marker_b, "{selector}");
        let stats = shared.stats();
        assert_eq!(
            stats.body_fetches, 2,
            "identical credential props must not collapse injected-io isolation: {selector}"
        );
        assert_eq!(stats.misses, 2, "{selector}");
    }
}

#[tokio::test]
async fn p1_commit_through_update_table_visible_on_shared_handle() {
    let file_io = FileIO::new_with_memory();
    let loc1 = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc1)
        .await
        .expect("write v1");

    let (state, source) = mutable_pointer(loc1, "tok-1");
    let cache = Arc::new(TableMetadataCache::new());
    let arn = "arn:aws:s3tables:us-east-1:1:bucket/commit";
    let creds = Some("ctx-shared".to_string());
    let cat_a = catalog(
        arn,
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        creds.clone(),
        Arc::clone(&source),
    )
    .await;
    let cat_b = catalog(
        arn,
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        creds,
        source,
    )
    .await
    .with_commit_transport(ScriptedS3TablesCommitTransport::new([
        S3TablesCommitScript::Success,
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

    *state.lock().expect("pointer state") = (loc2.clone(), "tok-2".to_string());

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
async fn l005_publish_carries_service_version_token() {
    let file_io = FileIO::new_with_memory();
    let loc1 = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc1)
        .await
        .expect("write v1");

    let (state, source) = mutable_pointer(loc1, "tok-1");
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/tok",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        Some("ctx".to_string()),
        source,
    )
    .await
    .with_commit_transport(ScriptedS3TablesCommitTransport::new([
        S3TablesCommitScript::SuccessToken("tok-2".to_string()),
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

    *state.lock().expect("pointer state") = (loc2.clone(), "tok-9".to_string());
    cat.load_table(&t)
        .await
        .expect("load under a different service version");
    let stats = cache.stats();
    assert_eq!(
        stats.misses, 2,
        "the published tok-2 entry must fail closed against service tok-9"
    );
    assert_eq!(stats.body_fetches, 2);

    cat.load_table(&t).await.expect("warm re-armed load");
    assert_eq!(
        cache.stats().hits,
        2,
        "commit base refresh + re-armed warm load"
    );
}

#[tokio::test]
async fn l002_register_table_is_unsupported_on_s3tables() {
    let file_io = FileIO::new_with_memory();
    let (_state, source) = mutable_pointer("memory://wh/t/metadata/v1.metadata.json", "tok");
    let cat = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/reg",
        &file_io,
        None,
        None,
        None,
        source,
    )
    .await;
    let err = cat
        .register_table(&ident("t"), "memory://wh/t/metadata/v1.metadata.json".to_string())
        .await
        .expect_err("register must fail");
    assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
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

    let (state, source) = mutable_pointer(loc1, "tok-1");
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/inv",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        Some("ctx".to_string()),
        source,
    )
    .await;
    let t = ident("t");

    cat.load_table(&t).await.expect("seed loc1 entry");
    *state.lock().expect("pointer state") = (loc2.to_string(), "tok-2".to_string());
    cat.load_table(&t).await.expect("seed loc2 entry");

    cat.invalidate_table(&t).await.expect("invalidate");
    cache.run_pending_tasks().await;

    let reloaded = cat.load_table(&t).await.expect("load after invalidate");
    assert_eq!(reloaded.metadata().location(), "memory://wh/t-moved");
    let stats = cache.stats();
    assert_eq!(
        stats.misses, 3,
        "the evicted current location must miss again"
    );
    assert_eq!(stats.body_fetches, 3);
    cache.run_pending_tasks().await;
    assert_eq!(cache.len(), 2, "loc1 entry stays; only the current location evicts");
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
    *state.lock().expect("pointer state") = Some((loc.to_string(), "tok".to_string()));
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/invfail",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        Some("ctx".to_string()),
        source,
    )
    .await;
    let t = ident("t");

    cat.load_table(&t).await.expect("seed load");
    *state.lock().expect("pointer state") = None;

    let err = cat
        .invalidate_table(&t)
        .await
        .expect_err("pointer failure must surface");
    assert_eq!(err.kind(), ErrorKind::Unexpected);

    *state.lock().expect("pointer state") = Some((loc.to_string(), "tok".to_string()));
    cat.load_table(&t).await.expect("load after failed invalidate");
    assert_eq!(
        cache.stats().hits, 1,
        "a failed pointer fetch must not evict the known entry"
    );
    assert_eq!(cache.stats().body_fetches, 1);
}

#[tokio::test]
async fn l3_drop_table_evicts_last_known_location() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write");

    let (_state, source) = mutable_pointer(loc, "tok");
    let dropped = Arc::new(Mutex::new(Vec::<String>::new()));
    let dropped_seen = Arc::clone(&dropped);
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/drop",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        Some("ctx".to_string()),
        source,
    )
    .await
    .with_drop_source(Arc::new(move |ident| {
        dropped_seen
            .lock()
            .expect("drop log")
            .push(ident.name().to_string());
        Ok(())
    }));
    let t = ident("t");

    cat.load_table(&t).await.expect("seed load");
    cat.drop_table(&t).await.expect("drop");
    cache.run_pending_tasks().await;
    assert_eq!(dropped.lock().expect("drop log").as_slice(), ["t"]);
    assert_eq!(
        cache.len(),
        0,
        "a successful drop must evict the last known location"
    );
}

#[tokio::test]
async fn l3_drop_failure_keeps_cache_entry() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write");

    let (_state, source) = mutable_pointer(loc, "tok");
    let cache = Arc::new(TableMetadataCache::new());
    let cat = catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/dropfail",
        &file_io,
        Some(Arc::clone(&cache)),
        None,
        Some("ctx".to_string()),
        source,
    )
    .await
    .with_drop_source(Arc::new(|_| {
        Err(Error::new(ErrorKind::Unexpected, "delete failed"))
    }));
    let t = ident("t");

    cat.load_table(&t).await.expect("seed load");
    cat.drop_table(&t).await.expect_err("drop must fail");
    cat.load_table(&t).await.expect("load after failed drop");
    assert_eq!(
        cache.stats().hits, 1,
        "a failed drop must not evict the entry"
    );
}

async fn builder_catalog(
    arn: &str,
    props_extra: &[(&str, &str)],
    shared: &Arc<TableMetadataCache>,
    file_io: &FileIO,
    source: PointerFn,
) -> S3TablesCatalog {
    S3TablesCatalogBuilder::default()
        .with_table_metadata_cache(Arc::clone(shared))
        .load("cat", builder_props(arn, props_extra))
        .await
        .expect("builder load")
        .with_file_io_for_tests(file_io.clone())
        .with_pointer_source(source)
}

#[tokio::test]
async fn p4_identical_credential_props_separate_by_identity_alone() {
    let file_io = FileIO::new_with_memory();
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write");
    let (_state, source) = mutable_pointer(loc, "tok");
    let shared = Arc::new(TableMetadataCache::new());
    let creds = [("aws_access_key_id", "SHARED-AKID")];
    let t = ident("t");

    let cat_a = builder_catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/a",
        &creds,
        &shared,
        &file_io,
        Arc::clone(&source),
    )
    .await;
    let cat_b = builder_catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/b",
        &creds,
        &shared,
        &file_io,
        Arc::clone(&source),
    )
    .await;
    let cat_a2 = builder_catalog(
        "arn:aws:s3tables:us-east-1:1:bucket/a",
        &creds,
        &shared,
        &file_io,
        source,
    )
    .await;

    cat_a.load_table(&t).await.expect("load a");
    cat_b.load_table(&t).await.expect("load b");
    cat_a2.load_table(&t).await.expect("load a2");

    let stats = shared.stats();
    assert_eq!(
        stats.misses, 2,
        "identical credentials must still isolate by catalog identity"
    );
    assert_eq!(stats.body_fetches, 2);
    assert_eq!(stats.hits, 1, "same identity + same context shares");
}

#[tokio::test]
async fn p9_builder_load_without_cache_rereads_every_load() {
    let loc = "memory://wh/t/metadata/v1.metadata.json";
    let file_io = FileIO::new_with_memory();
    sample_metadata("memory://wh/t")
        .write_to(&file_io, loc)
        .await
        .expect("write v1");
    let (_state, source) = mutable_pointer(loc, "tok");

    let cat = S3TablesCatalogBuilder::default()
        .load(
            "cat",
            builder_props("arn:aws:s3tables:us-east-1:1:bucket/raw", &[]),
        )
        .await
        .expect("builder load")
        .with_file_io_for_tests(file_io.clone())
        .with_pointer_source(source);
    let t = ident("t");

    cat.load_table(&t).await.expect("load 1");
    assert!(cat.table_metadata_cache.is_none());

    sample_metadata("memory://wh/t-rewritten")
        .write_to(&file_io, loc)
        .await
        .expect("rewrite");
    let second = cat.load_table(&t).await.expect("load 2");
    assert_eq!(
        second.metadata().location(),
        "memory://wh/t-rewritten",
        "a builder-loaded catalog without cache handles re-reads the body"
    );
}
