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
async fn l1_shared_credential_injected_factories_still_isolate() {
    for (selector, marker_a, marker_b) in [
        (
            "aws_access_key_id",
            "memory://wh/t-akid-a",
            "memory://wh/t-akid-b",
        ),
        (
            "profile_name",
            "memory://wh/t-prof-a",
            "memory://wh/t-prof-b",
        ),
    ] {
        let loc = "memory://wh/t/metadata/v1.metadata.json";
        let (_state, source) = mutable_pointer(loc, Some("vid"));
        let shared = Arc::new(TableMetadataCache::new());
        let t = ident("t");
        let props = builder_props(Some("shared-cat"), "memory://wh", &[(
            selector,
            "SHARED-CRED",
        )]);

        let cat_a = GlueCatalogBuilder::default()
            .with_table_metadata_cache(Arc::clone(&shared))
            .with_storage_factory(Arc::new(MemoryStorageFactory))
            .load("glue-a", props.clone())
            .await
            .expect("load catalog a")
            .with_pointer_source(Arc::clone(&source));
        let cat_b = GlueCatalogBuilder::default()
            .with_table_metadata_cache(Arc::clone(&shared))
            .with_storage_factory(Arc::new(MemoryStorageFactory))
            .load("glue-b", props)
            .await
            .expect("load catalog b")
            .with_pointer_source(source);

        sample_metadata(marker_a)
            .write_to(&cat_a.file_io(), loc)
            .await
            .expect("write a body");
        sample_metadata(marker_b)
            .write_to(&cat_b.file_io(), loc)
            .await
            .expect("write b body");

        let table_a = cat_a.load_table(&t).await.expect("load a");
        let table_b = cat_b.load_table(&t).await.expect("load b");
        assert_eq!(table_a.metadata().location(), marker_a, "{selector}");
        assert_eq!(table_b.metadata().location(), marker_b, "{selector}");
        let stats = shared.stats();
        assert_eq!(
            stats.body_fetches, 2,
            "identical credential props must not collapse injected-factory isolation: {selector}"
        );
        assert_eq!(stats.misses, 2, "{selector}");
    }
}
