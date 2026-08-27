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

//! Pin S3 Tables `register_table` as a dated service gap (F-9 / row R126).
//! The method returns before any AWS call, so this test needs no live service.

use std::collections::HashMap;

use iceberg::{Catalog, CatalogBuilder, ErrorKind, TableIdent};
use iceberg_catalog_s3tables::{S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN, S3TablesCatalogBuilder};

async fn test_catalog() -> iceberg::Result<iceberg_catalog_s3tables::S3TablesCatalog> {
    let config = aws_sdk_s3tables::Config::builder()
        .behavior_version(aws_sdk_s3tables::config::BehaviorVersion::latest())
        .region(aws_sdk_s3tables::config::Region::new("us-east-1"))
        .credentials_provider(aws_sdk_s3tables::config::Credentials::new(
            "test", "test", None, None, "test",
        ))
        .build();
    let client = aws_sdk_s3tables::Client::from_conf(config);
    S3TablesCatalogBuilder::default()
        .with_client(client)
        .load(
            "s3tables",
            HashMap::from([(
                S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN.to_string(),
                "arn:aws:s3tables:us-east-1:123456789012:bucket/test".to_string(),
            )]),
        )
        .await
}

#[tokio::test]
async fn register_table_is_a_dated_s3_tables_service_gap() {
    let catalog = test_catalog()
        .await
        .expect("catalog load must not need a live AWS call");
    let ident = TableIdent::from_strs(["ns", "t"]).expect("ident");
    let err = catalog
        .register_table(
            &ident,
            "s3://elsewhere/metadata/00000-deadbeef.metadata.json".to_string(),
        )
        .await
        .expect_err("S3 Tables has no register API");
    assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
    let message = err.message();
    assert!(
        message.contains("no register-by-metadata-location"),
        "got {message}"
    );
    assert!(
        message.contains("Iceberg REST register endpoint"),
        "message must name the missing REST register mapping, got {message}"
    );
    assert!(
        message.contains("UpdateTableMetadataLocation"),
        "message must name the warehouse-URI API, got {message}"
    );
    assert!(
        message.contains("R126"),
        "message must cite row R126, got {message}"
    );
    assert!(
        !message.contains("not supported yet"),
        "must not read as a temporary stub, got {message}"
    );
}
