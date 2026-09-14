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

use super::*;

async fn build_glue_catalog(name: Option<&str>, props: HashMap<String, String>) -> GlueCatalog {
    let config = GlueCatalogConfig {
        name: name.map(str::to_string),
        uri: None,
        catalog_id: None,
        warehouse: "s3://example-bucket/warehouse".to_string(),
        props,
    };
    GlueCatalog::new(config, None).await.unwrap()
}

#[tokio::test]
async fn test_name_and_properties_return_config() {
    let catalog = build_glue_catalog(
        Some("glue_cat"),
        HashMap::from([("region_name".to_string(), "us-east-1".to_string())]),
    )
    .await;

    assert_eq!(catalog.name(), "glue_cat");
    assert_eq!(
        catalog.properties().get("region_name").map(String::as_str),
        Some("us-east-1")
    );
}

#[tokio::test]
async fn test_name_defaults_to_sentinel_when_unset() {
    let catalog = build_glue_catalog(None, HashMap::new()).await;
    assert_eq!(catalog.name(), UNNAMED_CATALOG);
}

#[tokio::test]
async fn test_invalidate_defaults_are_noops() {
    let catalog = build_glue_catalog(Some("glue_cat"), HashMap::new()).await;
    let ident = TableIdent::new(NamespaceIdent::new("ns".to_string()), "t".to_string());
    catalog.invalidate_table(&ident).await.unwrap();
    catalog.invalidate_view(&ident).await.unwrap();
}

const SECRET: &str = "SECRET_DO_NOT_LEAK";

#[test]
fn test_config_debug_redacts_secret_prop_values() {
    let config = GlueCatalogConfig {
        name: Some("glue_cat".to_string()),
        uri: None,
        catalog_id: None,
        warehouse: "s3://example-bucket/warehouse".to_string(),
        props: HashMap::from([
            ("aws_secret_access_key".to_string(), SECRET.to_string()),
            ("aws_session_token".to_string(), SECRET.to_string()),
            ("region_name".to_string(), "us-east-1".to_string()),
        ]),
    };

    let debug = format!("{config:?}");

    assert!(
        !debug.contains(SECRET),
        "GlueCatalogConfig Debug leaked a secret value: {debug}"
    );
    assert!(debug.contains("***"), "expected redaction marker: {debug}");
    for key in ["aws_secret_access_key", "aws_session_token"] {
        assert!(debug.contains(key), "secret key `{key}` dropped: {debug}");
    }
    assert!(
        debug.contains("us-east-1"),
        "non-secret value must stay visible: {debug}"
    );
}

#[tokio::test]
async fn test_catalog_debug_redacts_secret_prop_values() {
    let catalog = build_glue_catalog(
        Some("glue_cat"),
        HashMap::from([("aws_secret_access_key".to_string(), SECRET.to_string())]),
    )
    .await;

    let debug = format!("{catalog:?}");

    assert!(
        !debug.contains(SECRET),
        "GlueCatalog Debug leaked a secret value: {debug}"
    );
    assert!(
        debug.contains("aws_secret_access_key"),
        "key dropped: {debug}"
    );
    assert!(debug.contains("***"), "expected redaction marker: {debug}");
}

#[test]
fn test_namespace_not_empty_error_maps_to_namespace_not_empty() {
    let err = namespace_not_empty_error("db1");
    assert_eq!(err.kind(), iceberg::ErrorKind::NamespaceNotEmpty);
    assert!(
        !err.retryable(),
        "a refused non-empty drop is not retryable"
    );
    assert!(
        err.message().contains("db1"),
        "message dropped db name: {err}"
    );
    assert!(
        err.message().contains("is not empty"),
        "message not preserved: {err}"
    );
}
