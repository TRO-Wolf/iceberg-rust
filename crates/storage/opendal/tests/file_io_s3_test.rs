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

//! Integration tests for FileIO S3.
//!
//! These tests assume Docker containers are started externally via `make docker-up`.
//! Each test uses unique file paths based on module path to avoid conflicts.
#[cfg(feature = "opendal-s3")]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use async_trait::async_trait;
    use iceberg::io::{
        FileIO, FileIOBuilder, S3_ACCESS_KEY_ID, S3_ENDPOINT, S3_REGION, S3_SECRET_ACCESS_KEY,
    };
    use iceberg::maintenance::DeleteOrphanFiles;
    use iceberg::memory::MemoryCatalogBuilder;
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
    use iceberg_storage_opendal::{CustomAwsCredentialLoader, OpenDalStorageFactory};
    use iceberg_test_utils::{get_minio_endpoint, normalize_test_name_with_parts, set_up};
    use reqsign::{AwsCredential, AwsCredentialLoad};
    use reqwest::Client;

    async fn get_file_io() -> FileIO {
        set_up();

        let minio_endpoint = get_minio_endpoint();

        FileIOBuilder::new(Arc::new(OpenDalStorageFactory::S3 {
            configured_scheme: "s3".to_string(),
            customized_credential_load: None,
        }))
        .with_props(vec![
            (S3_ENDPOINT, minio_endpoint),
            (S3_ACCESS_KEY_ID, "admin".to_string()),
            (S3_SECRET_ACCESS_KEY, "password".to_string()),
            (S3_REGION, "us-east-1".to_string()),
        ])
        .build()
    }

    #[tokio::test]
    async fn test_file_io_s3_exists() {
        let file_io = get_file_io().await;
        assert!(!file_io.exists("s3://bucket2/any").await.unwrap());
        assert!(file_io.exists("s3://bucket1/").await.unwrap());
    }

    #[tokio::test]
    async fn test_file_io_s3_output() {
        let file_io = get_file_io().await;
        // Use unique file path based on module path to avoid conflicts
        let output_path = format!(
            "s3://bucket1/{}",
            normalize_test_name_with_parts!("test_file_io_s3_output")
        );
        // Clean up from any previous test runs
        let _ = file_io.delete(&output_path).await;
        assert!(!file_io.exists(&output_path).await.unwrap());
        let output_file = file_io.new_output(&output_path).unwrap();
        {
            output_file.write("123".into()).await.unwrap();
        }
        assert!(file_io.exists(&output_path).await.unwrap());
    }

    #[tokio::test]
    async fn test_file_io_s3_list_bucket_root() {
        let file_io = get_file_io().await;
        let tag = normalize_test_name_with_parts!("test_file_io_s3_list_bucket_root");
        let file_path = format!("s3://bucket1/{tag}");
        let output_file = file_io.new_output(&file_path).unwrap();
        {
            output_file.write("root".into()).await.unwrap();
        }
        assert!(file_io.exists(&file_path).await.unwrap());

        let bare = file_io
            .list("s3://bucket1")
            .await
            .expect("list the bare bucket root");
        let slash = file_io
            .list("s3://bucket1/")
            .await
            .expect("list the slash bucket root");

        for location in bare.iter().map(|f| f.location.as_str()) {
            assert!(
                location.starts_with("s3://bucket1/"),
                "bare-bucket list must re-prefix entries, got {location}"
            );
        }
        assert!(bare.iter().any(|f| f.location == file_path));
        assert!(slash.iter().any(|f| f.location == file_path));

        let bare_set: HashSet<&str> = bare.iter().map(|f| f.location.as_str()).collect();
        let slash_set: HashSet<&str> = slash.iter().map(|f| f.location.as_str()).collect();
        assert_eq!(
            bare_set, slash_set,
            "bare and slash bucket roots must list the same set"
        );
    }

    #[tokio::test]
    async fn test_file_io_s3_delete_orphan_files_bucket_root() {
        let file_io = get_file_io().await;
        let tag =
            normalize_test_name_with_parts!("test_file_io_s3_delete_orphan_files_bucket_root");

        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(OpenDalStorageFactory::S3 {
                configured_scheme: "s3".to_string(),
                customized_credential_load: None,
            }))
            .load(
                "memory",
                HashMap::from([
                    ("warehouse".to_string(), "s3://bucket1".to_string()),
                    (S3_ENDPOINT.to_string(), get_minio_endpoint()),
                    (S3_ACCESS_KEY_ID.to_string(), "admin".to_string()),
                    (S3_SECRET_ACCESS_KEY.to_string(), "password".to_string()),
                    (S3_REGION.to_string(), "us-east-1".to_string()),
                ]),
            )
            .await
            .expect("load s3 memory catalog");

        let schema = Schema::builder()
            .with_fields(vec![Arc::new(NestedField::required(
                1,
                "x",
                Type::Primitive(PrimitiveType::Long),
            ))])
            .build()
            .expect("build schema");

        let bare_ns = NamespaceIdent::new(format!("ns_root_{tag}"));
        catalog
            .create_namespace(&bare_ns, HashMap::new())
            .await
            .expect("create namespace for the bucket-root table");
        let bare_table = catalog
            .create_table(
                &bare_ns,
                TableCreation::builder()
                    .name("t".to_string())
                    .schema(schema.clone())
                    .location("s3://bucket1".to_string())
                    .build(),
            )
            .await
            .expect("create the bucket-root table");
        let bare_metadata_location = bare_table
            .metadata_location()
            .expect("metadata location")
            .to_string();
        assert!(
            bare_metadata_location.starts_with("s3://bucket1/metadata/"),
            "bucket-root table metadata must live under the bucket root, got {bare_metadata_location}"
        );

        let root_orphan = format!("s3://bucket1/{tag}-orphan.dat");
        file_io
            .new_output(&root_orphan)
            .unwrap()
            .write("x".into())
            .await
            .unwrap();

        let recorded = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let sink = Arc::clone(&recorded);
        let result = DeleteOrphanFiles::new(bare_table)
            .older_than(i64::MAX)
            .delete_with(
                move |path: String| -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = iceberg::Result<()>> + Send>,
                > {
                    sink.lock().expect("recorder").push(path);
                    Box::pin(async { Ok(()) })
                },
            )
            .execute()
            .await
            .expect("orphan sweep over a bucket-root table");

        assert!(
            result.delete_failures.is_empty(),
            "{:?}",
            result.delete_failures
        );
        let mut root_orphans = result.orphan_file_locations.clone();
        root_orphans.sort();
        let mut recorded_paths = recorded.lock().expect("recorder").clone();
        recorded_paths.sort();
        assert_eq!(
            root_orphans, recorded_paths,
            "the recorded delete set is exactly the orphan set"
        );
        assert!(
            root_orphans.iter().any(|o| o == &root_orphan),
            "planted bucket-root orphan must be swept: {root_orphans:?}"
        );
        assert!(
            !root_orphans.iter().any(|o| o == &bare_metadata_location),
            "the live metadata file must never be swept"
        );

        let ctl_ns = NamespaceIdent::new(format!("ns_ctl_{tag}"));
        catalog
            .create_namespace(&ctl_ns, HashMap::new())
            .await
            .expect("create namespace for the control table");
        let ctl_location = format!("s3://bucket1/{tag}-ctl/tbl");
        let ctl_table = catalog
            .create_table(
                &ctl_ns,
                TableCreation::builder()
                    .name("t".to_string())
                    .schema(schema)
                    .location(ctl_location.clone())
                    .build(),
            )
            .await
            .expect("create the nested control table");
        let ctl_orphan = format!("{ctl_location}/{tag}-orphan.dat");
        file_io
            .new_output(&ctl_orphan)
            .unwrap()
            .write("x".into())
            .await
            .unwrap();

        let ctl_result = DeleteOrphanFiles::new(ctl_table)
            .older_than(i64::MAX)
            .execute()
            .await
            .expect("orphan sweep over the nested control table");
        assert!(ctl_result.delete_failures.is_empty());
        assert_eq!(
            ctl_result.orphan_file_locations,
            vec![ctl_orphan],
            "a nested table sweeps exactly its unreferenced files"
        );
    }

    #[tokio::test]
    async fn test_file_io_s3_input() {
        let file_io = get_file_io().await;
        // Use unique file path based on module path to avoid conflicts
        let file_path = format!(
            "s3://bucket1/{}",
            normalize_test_name_with_parts!("test_file_io_s3_input")
        );
        let output_file = file_io.new_output(&file_path).unwrap();
        {
            output_file.write("test_input".into()).await.unwrap();
        }

        let input_file = file_io.new_input(&file_path).unwrap();

        {
            let buffer = input_file.read().await.unwrap();
            assert_eq!(buffer, "test_input".as_bytes());
        }
    }

    // Mock credential loader for testing
    struct MockCredentialLoader {
        credential: Option<AwsCredential>,
    }

    impl MockCredentialLoader {
        fn new(credential: Option<AwsCredential>) -> Self {
            Self { credential }
        }

        fn new_minio() -> Self {
            Self::new(Some(AwsCredential {
                access_key_id: "admin".to_string(),
                secret_access_key: "password".to_string(),
                session_token: None,
                expires_in: None,
            }))
        }
    }

    #[async_trait]
    impl AwsCredentialLoad for MockCredentialLoader {
        async fn load_credential(&self, _client: Client) -> anyhow::Result<Option<AwsCredential>> {
            Ok(self.credential.clone())
        }
    }

    #[test]
    fn test_custom_aws_credential_loader_instantiation() {
        // Test creating CustomAwsCredentialLoader with mock loader
        let mock_loader = MockCredentialLoader::new_minio();
        let custom_loader = CustomAwsCredentialLoader::new(Arc::new(mock_loader));

        // Test that the loader can be used in FileIOBuilder with OpenDalStorageFactory
        let _builder = FileIOBuilder::new(Arc::new(OpenDalStorageFactory::S3 {
            configured_scheme: "s3".to_string(),
            customized_credential_load: Some(custom_loader),
        }))
        .with_props(vec![
            (S3_ENDPOINT, "http://localhost:9000".to_string()),
            ("bucket", "test-bucket".to_string()),
            (S3_REGION, "us-east-1".to_string()),
        ]);
    }

    #[tokio::test]
    async fn test_s3_with_custom_credential_loader_integration() {
        let _file_io = get_file_io().await;

        // Create a mock credential loader
        let mock_loader = MockCredentialLoader::new_minio();
        let custom_loader = CustomAwsCredentialLoader::new(Arc::new(mock_loader));

        let minio_endpoint = get_minio_endpoint();

        // Build FileIO with custom credential loader via OpenDalStorageFactory
        let file_io_with_custom_creds = FileIOBuilder::new(Arc::new(OpenDalStorageFactory::S3 {
            configured_scheme: "s3".to_string(),
            customized_credential_load: Some(custom_loader),
        }))
        .with_props(vec![
            (S3_ENDPOINT, minio_endpoint),
            (S3_REGION, "us-east-1".to_string()),
        ])
        .build();

        // Test that the FileIO was built successfully with the custom loader
        match file_io_with_custom_creds.exists("s3://bucket1/any").await {
            Ok(_) => {}
            Err(e) => panic!("Failed to check existence of bucket: {e}"),
        }
    }

    #[tokio::test]
    async fn test_s3_with_custom_credential_loader_integration_failure() {
        let _file_io = get_file_io().await;

        // Create a mock credential loader with no credentials
        let mock_loader = MockCredentialLoader::new(None);
        let custom_loader = CustomAwsCredentialLoader::new(Arc::new(mock_loader));

        let minio_endpoint = get_minio_endpoint();

        // Build FileIO with custom credential loader via OpenDalStorageFactory
        let file_io_with_custom_creds = FileIOBuilder::new(Arc::new(OpenDalStorageFactory::S3 {
            configured_scheme: "s3".to_string(),
            customized_credential_load: Some(custom_loader),
        }))
        .with_props(vec![
            (S3_ENDPOINT, minio_endpoint),
            (S3_REGION, "us-east-1".to_string()),
        ])
        .build();

        // Test that the FileIO was built successfully with the custom loader
        match file_io_with_custom_creds.exists("s3://bucket1/any").await {
            Ok(_) => panic!(
                "Expected error, but got Ok - the credential loader should fail to provide valid credentials"
            ),
            Err(e) => {
                assert!(
                    e.to_string()
                        .contains("no valid credential found and anonymous access is not allowed")
                );
            }
        }
    }
}
