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

use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::transaction::{ApplyTransactionAction, Transaction};

use super::*;

const SECRET: &str = "SECRET_DO_NOT_LEAK";

fn config_with_secret_props() -> S3TablesCatalogConfig {
    S3TablesCatalogConfig {
        name: Some("s3t_cat".to_string()),
        table_bucket_arn: "arn:aws:s3tables:us-east-1:123456789012:bucket/example".to_string(),
        endpoint_url: None,
        client: None,
        props: HashMap::from([
            ("aws_secret_access_key".to_string(), SECRET.to_string()),
            ("aws_session_token".to_string(), SECRET.to_string()),
            ("region_name".to_string(), "us-east-1".to_string()),
        ]),
    }
}

#[test]
fn test_config_debug_redacts_secret_prop_values() {
    let config = config_with_secret_props();

    let debug = format!("{config:?}");

    assert!(
        !debug.contains(SECRET),
        "S3TablesCatalogConfig Debug leaked a secret value: {debug}"
    );
    assert!(debug.contains("***"), "expected redaction marker: {debug}");
    for key in ["aws_secret_access_key", "aws_session_token"] {
        assert!(debug.contains(key), "secret key `{key}` dropped: {debug}");
    }
    assert!(
        debug.contains("us-east-1") && debug.contains("s3t_cat"),
        "non-secret fields must stay visible: {debug}"
    );
}

#[tokio::test]
async fn test_catalog_debug_redacts_secret_prop_values() {
    let catalog = S3TablesCatalog::new(config_with_secret_props(), None)
        .await
        .expect("build S3TablesCatalog offline");

    let debug = format!("{catalog:?}");

    assert!(
        !debug.contains(SECRET),
        "S3TablesCatalog Debug leaked a secret value: {debug}"
    );
    assert!(
        debug.contains("aws_secret_access_key"),
        "key dropped: {debug}"
    );
    assert!(debug.contains("***"), "expected redaction marker: {debug}");
}

async fn load_s3tables_catalog_from_env() -> Result<Option<S3TablesCatalog>> {
    let table_bucket_arn = match std::env::var("TABLE_BUCKET_ARN").ok() {
        Some(table_bucket_arn) => table_bucket_arn,
        None => return Ok(None),
    };

    let config = S3TablesCatalogConfig {
        name: None,
        table_bucket_arn,
        endpoint_url: None,
        client: None,
        props: HashMap::new(),
    };

    Ok(Some(S3TablesCatalog::new(config, None).await?))
}

#[tokio::test]
async fn test_s3tables_list_namespace() {
    let catalog = match load_s3tables_catalog_from_env().await {
        Ok(Some(catalog)) => catalog,
        Ok(None) => return,
        Err(e) => panic!("Error loading catalog: {e}"),
    };

    let namespaces = catalog.list_namespaces(None).await.unwrap();
    assert!(!namespaces.is_empty());
}

#[tokio::test]
async fn test_s3tables_list_tables() {
    let catalog = match load_s3tables_catalog_from_env().await {
        Ok(Some(catalog)) => catalog,
        Ok(None) => return,
        Err(e) => panic!("Error loading catalog: {e}"),
    };

    let tables = catalog
        .list_tables(&NamespaceIdent::new("aws_s3_metadata".to_string()))
        .await
        .unwrap();
    assert!(!tables.is_empty());
}

#[tokio::test]
async fn test_s3tables_load_table() {
    let catalog = match load_s3tables_catalog_from_env().await {
        Ok(Some(catalog)) => catalog,
        Ok(None) => return,
        Err(e) => panic!("Error loading catalog: {e}"),
    };

    let table = catalog
        .load_table(&TableIdent::new(
            NamespaceIdent::new("aws_s3_metadata".to_string()),
            "query_storage_metadata".to_string(),
        ))
        .await
        .unwrap();
    println!("{table:?}");
}

#[tokio::test]
async fn test_s3tables_create_delete_namespace() {
    let catalog = match load_s3tables_catalog_from_env().await {
        Ok(Some(catalog)) => catalog,
        Ok(None) => return,
        Err(e) => panic!("Error loading catalog: {e}"),
    };

    let namespace = NamespaceIdent::new("test_s3tables_create_delete_namespace".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    assert!(catalog.namespace_exists(&namespace).await.unwrap());
    catalog.drop_namespace(&namespace).await.unwrap();
    assert!(!catalog.namespace_exists(&namespace).await.unwrap());
}

#[tokio::test]
async fn test_s3tables_create_delete_table() {
    let catalog = match load_s3tables_catalog_from_env().await {
        Ok(Some(catalog)) => catalog,
        Ok(None) => return,
        Err(e) => panic!("Error loading catalog: {e}"),
    };

    let creation = {
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "bar", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        TableCreation::builder()
            .name("test_s3tables_create_delete_table".to_string())
            .properties(HashMap::new())
            .schema(schema)
            .build()
    };

    let namespace = NamespaceIdent::new("test_s3tables_create_delete_table".to_string());
    let table_ident = TableIdent::new(
        namespace.clone(),
        "test_s3tables_create_delete_table".to_string(),
    );
    catalog.drop_namespace(&namespace).await.ok();
    catalog.drop_table(&table_ident).await.ok();

    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    catalog.create_table(&namespace, creation).await.unwrap();
    assert!(catalog.table_exists(&table_ident).await.unwrap());
    catalog.drop_table(&table_ident).await.unwrap();
    assert!(!catalog.table_exists(&table_ident).await.unwrap());
    catalog.drop_namespace(&namespace).await.unwrap();
}

#[tokio::test]
async fn test_s3tables_update_table() {
    let catalog = match load_s3tables_catalog_from_env().await {
        Ok(Some(catalog)) => catalog,
        Ok(None) => return,
        Err(e) => panic!("Error loading catalog: {e}"),
    };

    let namespace = NamespaceIdent::new("test_s3tables_update_table".to_string());
    let table_ident =
        TableIdent::new(namespace.clone(), "test_s3tables_update_table".to_string());

    catalog.drop_table(&table_ident).await.ok();
    catalog.drop_namespace(&namespace).await.ok();

    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();

    let creation = {
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "bar", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        TableCreation::builder()
            .name(table_ident.name().to_string())
            .properties(HashMap::new())
            .schema(schema)
            .build()
    };

    let table = catalog.create_table(&namespace, creation).await.unwrap();

    let tx = Transaction::new(&table);

    let original_metadata_location = table.metadata_location();

    let tx = tx
        .update_table_properties()
        .set("test_property".to_string(), "test_value".to_string())
        .apply(tx)
        .unwrap();

    let updated_table = tx.commit(&catalog).await.unwrap();

    assert_eq!(
        updated_table.metadata().properties().get("test_property"),
        Some(&"test_value".to_string())
    );

    assert_ne!(
        updated_table.metadata_location(),
        original_metadata_location,
        "Metadata location should be updated after commit"
    );

    let reloaded_table = catalog.load_table(&table_ident).await.unwrap();

    assert_eq!(
        reloaded_table.metadata().properties().get("test_property"),
        Some(&"test_value".to_string())
    );
    assert_eq!(
        reloaded_table.metadata_location(),
        updated_table.metadata_location(),
        "Reloaded table should have the same metadata location as the updated table"
    );
}

#[tokio::test]
async fn test_builder_load_missing_bucket_arn() {
    let builder = S3TablesCatalogBuilder::default();
    let result = builder.load("s3tables", HashMap::new()).await;

    assert!(result.is_err());
    if let Err(err) = result {
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(err.message(), "Table bucket ARN is required");
    }
}

#[tokio::test]
async fn test_builder_with_endpoint_url_ok() {
    let builder = S3TablesCatalogBuilder::default().with_endpoint_url("http://localhost:4566");

    let result = builder
        .load(
            "s3tables",
            HashMap::from([
                (
                    S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN.to_string(),
                    "arn:aws:s3tables:us-east-1:123456789012:bucket/test".to_string(),
                ),
                ("some_prop".to_string(), "some_value".to_string()),
            ]),
        )
        .await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_builder_with_client_ok() {
    use aws_config::BehaviorVersion;

    let sdk_config = aws_config::defaults(BehaviorVersion::latest()).load().await;
    let client = aws_sdk_s3tables::Client::new(&sdk_config);

    let builder = S3TablesCatalogBuilder::default().with_client(client);
    let result = builder
        .load(
            "s3tables",
            HashMap::from([(
                S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN.to_string(),
                "arn:aws:s3tables:us-east-1:123456789012:bucket/test".to_string(),
            )]),
        )
        .await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_builder_with_table_bucket_arn() {
    let test_arn = "arn:aws:s3tables:us-west-2:123456789012:bucket/test-bucket";
    let builder = S3TablesCatalogBuilder::default().with_table_bucket_arn(test_arn);

    let result = builder.load("s3tables", HashMap::new()).await;

    assert!(result.is_ok());
    let catalog = result.unwrap();
    assert_eq!(catalog.config.table_bucket_arn, test_arn);
}

#[tokio::test]
async fn test_builder_empty_table_bucket_arn_edge_cases() {
    let mut props = HashMap::new();
    props.insert(
        S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN.to_string(),
        "".to_string(),
    );

    let builder = S3TablesCatalogBuilder::default();
    let result = builder.load("s3tables", props).await;

    assert!(result.is_err());
    if let Err(err) = result {
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(err.message(), "Table bucket ARN is required");
    }
}

#[tokio::test]
async fn test_endpoint_url_property_overrides_builder_method() {
    let test_arn = "arn:aws:s3tables:us-west-2:123456789012:bucket/test-bucket";
    let builder_endpoint = "http://localhost:4566";
    let property_endpoint = "http://localhost:8080";

    let builder = S3TablesCatalogBuilder::default()
        .with_table_bucket_arn(test_arn)
        .with_endpoint_url(builder_endpoint);

    let mut props = HashMap::new();
    props.insert(
        S3TABLES_CATALOG_PROP_ENDPOINT_URL.to_string(),
        property_endpoint.to_string(),
    );

    let result = builder.load("s3tables", props).await;

    assert!(result.is_ok());
    let catalog = result.unwrap();

    assert_eq!(
        catalog.config.endpoint_url,
        Some(property_endpoint.to_string())
    );
    assert_ne!(
        catalog.config.endpoint_url,
        Some(builder_endpoint.to_string())
    );
}

#[tokio::test]
async fn test_endpoint_url_builder_method_only() {
    let test_arn = "arn:aws:s3tables:us-west-2:123456789012:bucket/test-bucket";
    let builder_endpoint = "http://localhost:4566";

    let builder = S3TablesCatalogBuilder::default()
        .with_table_bucket_arn(test_arn)
        .with_endpoint_url(builder_endpoint);

    let result = builder.load("s3tables", HashMap::new()).await;

    assert!(result.is_ok());
    let catalog = result.unwrap();

    assert_eq!(
        catalog.config.endpoint_url,
        Some(builder_endpoint.to_string())
    );
}

#[tokio::test]
async fn test_endpoint_url_property_only() {
    let test_arn = "arn:aws:s3tables:us-west-2:123456789012:bucket/test-bucket";
    let property_endpoint = "http://localhost:8080";

    let builder = S3TablesCatalogBuilder::default().with_table_bucket_arn(test_arn);

    let mut props = HashMap::new();
    props.insert(
        S3TABLES_CATALOG_PROP_ENDPOINT_URL.to_string(),
        property_endpoint.to_string(),
    );

    let result = builder.load("s3tables", props).await;

    assert!(result.is_ok());
    let catalog = result.unwrap();

    assert_eq!(
        catalog.config.endpoint_url,
        Some(property_endpoint.to_string())
    );
}

#[tokio::test]
async fn test_table_bucket_arn_property_overrides_builder_method() {
    let builder_arn = "arn:aws:s3tables:us-west-2:123456789012:bucket/builder-bucket";
    let property_arn = "arn:aws:s3tables:us-east-1:987654321098:bucket/property-bucket";

    let builder = S3TablesCatalogBuilder::default().with_table_bucket_arn(builder_arn);

    let mut props = HashMap::new();
    props.insert(
        S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN.to_string(),
        property_arn.to_string(),
    );

    let result = builder.load("s3tables", props).await;

    assert!(result.is_ok());
    let catalog = result.unwrap();

    assert_eq!(catalog.config.table_bucket_arn, property_arn);
    assert_ne!(catalog.config.table_bucket_arn, builder_arn);
}

#[tokio::test]
async fn test_table_bucket_arn_builder_method_only() {
    let builder_arn = "arn:aws:s3tables:us-west-2:123456789012:bucket/builder-bucket";

    let builder = S3TablesCatalogBuilder::default().with_table_bucket_arn(builder_arn);

    let result = builder.load("s3tables", HashMap::new()).await;

    assert!(result.is_ok());
    let catalog = result.unwrap();

    assert_eq!(catalog.config.table_bucket_arn, builder_arn);
}

#[tokio::test]
async fn test_table_bucket_arn_property_only() {
    let property_arn = "arn:aws:s3tables:us-east-1:987654321098:bucket/property-bucket";

    let builder = S3TablesCatalogBuilder::default();

    let mut props = HashMap::new();
    props.insert(
        S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN.to_string(),
        property_arn.to_string(),
    );

    let result = builder.load("s3tables", props).await;

    assert!(result.is_ok());
    let catalog = result.unwrap();

    assert_eq!(catalog.config.table_bucket_arn, property_arn);
}

#[tokio::test]
async fn test_builder_empty_name_validation() {
    let test_arn = "arn:aws:s3tables:us-west-2:123456789012:bucket/test-bucket";
    let builder = S3TablesCatalogBuilder::default().with_table_bucket_arn(test_arn);

    let result = builder.load("", HashMap::new()).await;

    assert!(result.is_err());
    if let Err(err) = result {
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(err.message(), "Catalog name cannot be empty");
    }
}

#[tokio::test]
async fn test_builder_whitespace_only_name_validation() {
    let test_arn = "arn:aws:s3tables:us-west-2:123456789012:bucket/test-bucket";
    let builder = S3TablesCatalogBuilder::default().with_table_bucket_arn(test_arn);

    let result = builder.load("   \t\n  ", HashMap::new()).await;

    assert!(result.is_err());
    if let Err(err) = result {
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(err.message(), "Catalog name cannot be empty");
    }
}

#[tokio::test]
async fn test_builder_name_validation_with_missing_arn() {
    let builder = S3TablesCatalogBuilder::default();

    let result = builder.load("", HashMap::new()).await;

    assert!(result.is_err());
    if let Err(err) = result {
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(err.message(), "Catalog name cannot be empty");
    }
}

#[tokio::test]
async fn test_name_and_properties_return_config() {
    let config = S3TablesCatalogConfig {
        name: Some("s3t_cat".to_string()),
        table_bucket_arn: "arn:aws:s3tables:us-east-1:123456789012:bucket/example".to_string(),
        endpoint_url: None,
        client: None,
        props: HashMap::from([("region_name".to_string(), "us-east-1".to_string())]),
    };
    let catalog = S3TablesCatalog::new(config, None).await.unwrap();

    assert_eq!(catalog.name(), "s3t_cat");
    assert_eq!(
        catalog.properties().get("region_name").map(String::as_str),
        Some("us-east-1")
    );
}

#[tokio::test]
async fn test_name_defaults_to_sentinel_when_unset() {
    let config = S3TablesCatalogConfig {
        name: None,
        table_bucket_arn: "arn:aws:s3tables:us-east-1:123456789012:bucket/example".to_string(),
        endpoint_url: None,
        client: None,
        props: HashMap::new(),
    };
    let catalog = S3TablesCatalog::new(config, None).await.unwrap();
    assert_eq!(catalog.name(), UNNAMED_CATALOG);
}

#[tokio::test]
async fn test_invalidate_table_without_cache_is_noop_and_view_is_noop() {
    let config = S3TablesCatalogConfig {
        name: Some("s3t_cat".to_string()),
        table_bucket_arn: "arn:aws:s3tables:us-east-1:123456789012:bucket/example".to_string(),
        endpoint_url: None,
        client: None,
        props: HashMap::new(),
    };
    let catalog = S3TablesCatalog::new(config, None).await.unwrap();
    let ident = TableIdent::new(NamespaceIdent::new("ns".to_string()), "t".to_string());
    catalog.invalidate_table(&ident).await.unwrap();
    catalog.invalidate_view(&ident).await.unwrap();
}

#[tokio::test]
async fn l001_footer_cache_default_off() {
    assert!(S3TablesCatalogBuilder::default().shared_footer_cache.is_none());
    let config = S3TablesCatalogConfig {
        name: Some("s3t_cat".to_string()),
        table_bucket_arn: "arn:aws:s3tables:us-east-1:123456789012:bucket/example".to_string(),
        endpoint_url: None,
        client: None,
        props: HashMap::new(),
    };
    let catalog = S3TablesCatalog::new(config, None).await.unwrap();
    assert!(catalog.shared_footer_cache.is_none());
}
