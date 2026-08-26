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
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use aws_sdk_s3tables::operation::create_table::CreateTableOutput;
use aws_sdk_s3tables::operation::get_namespace::GetNamespaceOutput;
use aws_sdk_s3tables::operation::get_table::GetTableOutput;
use aws_sdk_s3tables::operation::list_tables::ListTablesOutput;
use aws_sdk_s3tables::operation::update_table_metadata_location::UpdateTableMetadataLocationError;
use aws_sdk_s3tables::types::OpenTableFormat;
use iceberg::io::{FileIO, FileIOBuilder, StorageFactory};
use iceberg::spec::{TableMetadata, TableMetadataBuilder};
use iceberg::table::Table;
use iceberg::{
    Catalog, CatalogBuilder, CommitBaseLoadPlan, Error, ErrorKind, MetadataLocation, Namespace,
    NamespaceIdent, Result, TableCommit, TableCreation, TableIdent, UNNAMED_CATALOG,
    commit_base_conflict_error, plan_commit_base_load,
};
use iceberg_storage_opendal::OpenDalStorageFactory;

use crate::utils::create_sdk_config;

/// S3Tables table bucket ARN property
pub const S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN: &str = "table_bucket_arn";
/// S3Tables endpoint URL property
pub const S3TABLES_CATALOG_PROP_ENDPOINT_URL: &str = "endpoint_url";

/// S3Tables catalog configuration.
struct S3TablesCatalogConfig {
    /// Catalog name.
    name: Option<String>,
    /// Unlike other buckets, S3Tables bucket is not a physical bucket, but a virtual bucket
    /// that is managed by s3tables. We can't directly access the bucket with path like
    /// s3://{bucket_name}/{file_path}, all the operations are done with respect of the bucket
    /// ARN.
    table_bucket_arn: String,
    /// Endpoint URL for the catalog.
    endpoint_url: Option<String>,
    /// Optional pre-configured AWS SDK client for S3Tables.
    client: Option<aws_sdk_s3tables::Client>,
    /// Properties for the catalog. The available properties are:
    /// - `profile_name`: The name of the AWS profile to use.
    /// - `region_name`: The AWS region to use.
    /// - `aws_access_key_id`: The AWS access key ID to use.
    /// - `aws_secret_access_key`: The AWS secret access key to use.
    /// - `aws_session_token`: The AWS session token to use.
    props: HashMap<String, String>,
}

impl std::fmt::Debug for S3TablesCatalogConfig {
    /// Redact secret prop values. The AWS credentials in `props` flow into the `FileIO` this
    /// config backs, so a derived `Debug` prints them in clear. Redaction uses the canonical
    /// `iceberg::io::is_secret_prop_key`, so the secret-key list cannot drift per catalog.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let redacted_props: HashMap<&str, &str> = self
            .props
            .iter()
            .map(|(k, v)| {
                if iceberg::io::is_secret_prop_key(k) {
                    (k.as_str(), "***")
                } else {
                    (k.as_str(), v.as_str())
                }
            })
            .collect();

        f.debug_struct("S3TablesCatalogConfig")
            .field("name", &self.name)
            .field("table_bucket_arn", &self.table_bucket_arn)
            .field("endpoint_url", &self.endpoint_url)
            .field("client_configured", &self.client.is_some())
            .field("props", &redacted_props)
            .finish()
    }
}

/// Builder for [`S3TablesCatalog`].
#[derive(Debug)]
pub struct S3TablesCatalogBuilder {
    config: S3TablesCatalogConfig,
    storage_factory: Option<Arc<dyn StorageFactory>>,
}

/// Default builder for [`S3TablesCatalog`].
impl Default for S3TablesCatalogBuilder {
    fn default() -> Self {
        Self {
            config: S3TablesCatalogConfig {
                name: None,
                table_bucket_arn: "".to_string(),
                endpoint_url: None,
                client: None,
                props: HashMap::new(),
            },
            storage_factory: None,
        }
    }
}

/// Builder methods for [`S3TablesCatalog`].
impl S3TablesCatalogBuilder {
    /// Configure the catalog with a custom endpoint URL (useful for local testing/mocking).
    ///
    /// The `endpoint_url` property passed to `load()` overrides this value.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.config.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Configure the catalog with a pre-built AWS SDK client.
    pub fn with_client(mut self, client: aws_sdk_s3tables::Client) -> Self {
        self.config.client = Some(client);
        self
    }

    /// Configure the catalog with a table bucket ARN.
    ///
    /// The `table_bucket_arn` property passed to `load()` overrides this value.
    pub fn with_table_bucket_arn(mut self, table_bucket_arn: impl Into<String>) -> Self {
        self.config.table_bucket_arn = table_bucket_arn.into();
        self
    }
}

impl CatalogBuilder for S3TablesCatalogBuilder {
    type C = S3TablesCatalog;

    fn with_storage_factory(mut self, storage_factory: Arc<dyn StorageFactory>) -> Self {
        self.storage_factory = Some(storage_factory);
        self
    }

    fn load(
        mut self,
        name: impl Into<String>,
        props: HashMap<String, String>,
    ) -> impl Future<Output = Result<Self::C>> + Send {
        let catalog_name = name.into();
        self.config.name = Some(catalog_name.clone());

        if props.contains_key(S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN) {
            self.config.table_bucket_arn = props
                .get(S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN)
                .cloned()
                .unwrap_or_default();
        }

        if props.contains_key(S3TABLES_CATALOG_PROP_ENDPOINT_URL) {
            self.config.endpoint_url = props.get(S3TABLES_CATALOG_PROP_ENDPOINT_URL).cloned();
        }

        self.config.props = props
            .into_iter()
            .filter(|(k, _)| {
                k != S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN
                    && k != S3TABLES_CATALOG_PROP_ENDPOINT_URL
            })
            .collect();

        async move {
            if catalog_name.trim().is_empty() {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Catalog name cannot be empty",
                ))
            } else if self.config.table_bucket_arn.is_empty() {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Table bucket ARN is required",
                ))
            } else {
                S3TablesCatalog::new(self.config, self.storage_factory).await
            }
        }
    }
}

/// S3Tables catalog implementation.
#[derive(Debug)]
pub struct S3TablesCatalog {
    config: S3TablesCatalogConfig,
    s3tables_client: aws_sdk_s3tables::Client,
    file_io: FileIO,
}

impl S3TablesCatalog {
    /// Creates a new S3Tables catalog.
    async fn new(
        config: S3TablesCatalogConfig,
        storage_factory: Option<Arc<dyn StorageFactory>>,
    ) -> Result<Self> {
        let s3tables_client = if let Some(client) = config.client.clone() {
            client
        } else {
            let aws_config = create_sdk_config(&config.props, config.endpoint_url.clone()).await;
            aws_sdk_s3tables::Client::new(&aws_config)
        };

        let factory = storage_factory.unwrap_or_else(|| {
            Arc::new(OpenDalStorageFactory::S3 {
                configured_scheme: "s3".to_string(),
                customized_credential_load: None,
            })
        });
        let file_io = FileIOBuilder::new(factory)
            .with_props(&config.props)
            .build();

        Ok(Self {
            config,
            s3tables_client,
            file_io,
        })
    }

    /// GetTable for the service metadata pointer and version_token. Reads no object storage.
    async fn get_table_pointer(
        &self,
        table_ident: &TableIdent,
    ) -> Result<(
        String, /* metadata_location */
        String, /* version_token */
    )> {
        let req = self
            .s3tables_client
            .get_table()
            .table_bucket_arn(self.config.table_bucket_arn.clone())
            .namespace(table_ident.namespace().to_url_string())
            .name(table_ident.name());
        let resp: GetTableOutput = req.send().await.map_err(from_aws_sdk_error)?;

        // when a table is created, it's possible that the metadata location is not set.
        let metadata_location = resp.metadata_location().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!(
                    "Table {} does not have metadata location",
                    table_ident.name()
                ),
            )
        })?;
        Ok((metadata_location.to_string(), resp.version_token))
    }

    async fn load_table_with_version_token(
        &self,
        table_ident: &TableIdent,
    ) -> Result<(Table, String)> {
        let (metadata_location, version_token) = self.get_table_pointer(table_ident).await?;
        let metadata = TableMetadata::read_from(&self.file_io, &metadata_location).await?;

        let table = Table::builder()
            .identifier(table_ident.clone())
            .metadata(metadata)
            .metadata_location(metadata_location)
            .file_io(self.file_io.clone())
            .build()?;
        Ok((table, version_token))
    }

    /// Resolve the base table for a commit. Reuses a pre-loaded base when the service pointer
    /// still matches, which skips the S3 metadata parse.
    async fn resolve_commit_base(
        &self,
        table_ident: &TableIdent,
        commit: &mut TableCommit,
    ) -> Result<(Table, String /* version_token */)> {
        let (service_location, version_token) = self.get_table_pointer(table_ident).await?;
        let base_loc = commit.base_metadata_location().map(str::to_string);
        let provided = commit.take_base_table();
        let provided_loc = provided
            .as_ref()
            .and_then(|t| t.metadata_location().map(str::to_string));

        match plan_commit_base_load(
            &service_location,
            base_loc.as_deref(),
            provided_loc.as_deref(),
        ) {
            CommitBaseLoadPlan::ReuseProvided => {
                let provided = provided.ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "commit base-load plan is ReuseProvided but no base table was supplied",
                    )
                })?;
                // Rebind catalog FileIO + commit identifier (defense in depth vs forged base).
                let table = Table::builder()
                    .identifier(table_ident.clone())
                    .metadata(provided.metadata_ref())
                    .metadata_location(service_location)
                    .file_io(self.file_io.clone())
                    .build()?;
                Ok((table, version_token))
            }
            CommitBaseLoadPlan::Conflict => Err(commit_base_conflict_error(
                table_ident,
                base_loc.as_deref(),
                &service_location,
            )),
            CommitBaseLoadPlan::FullLoad => {
                let metadata = TableMetadata::read_from(&self.file_io, &service_location).await?;
                let table = Table::builder()
                    .identifier(table_ident.clone())
                    .metadata(metadata)
                    .metadata_location(service_location)
                    .file_io(self.file_io.clone())
                    .build()?;
                Ok((table, version_token))
            }
        }
    }
}

#[async_trait]
impl Catalog for S3TablesCatalog {
    /// Returns the catalog name given to [`CatalogBuilder::load`], or [`UNNAMED_CATALOG`].
    fn name(&self) -> &str {
        self.config.name.as_deref().unwrap_or(UNNAMED_CATALOG)
    }

    /// Returns the configuration properties supplied at construction.
    fn properties(&self) -> &HashMap<String, String> {
        &self.config.props
    }

    /// List namespaces from s3tables catalog.
    ///
    /// S3Tables has no nested namespaces, so a `parent` always returns an empty list.
    async fn list_namespaces(
        &self,
        parent: Option<&NamespaceIdent>,
    ) -> Result<Vec<NamespaceIdent>> {
        if parent.is_some() {
            return Ok(vec![]);
        }

        let mut result = Vec::new();
        let mut continuation_token = None;
        loop {
            let mut req = self
                .s3tables_client
                .list_namespaces()
                .table_bucket_arn(self.config.table_bucket_arn.clone());
            if let Some(token) = continuation_token {
                req = req.continuation_token(token);
            }
            let resp = req.send().await.map_err(from_aws_sdk_error)?;
            for ns in resp.namespaces() {
                result.push(NamespaceIdent::from_vec(ns.namespace().to_vec())?);
            }
            continuation_token = resp.continuation_token().map(|s| s.to_string());
            if continuation_token.is_none() {
                break;
            }
        }
        Ok(result)
    }

    /// Creates a new namespace. The `properties` parameter is ignored.
    ///
    /// S3Tables namespace names are 3 to 63 characters long. They use only lowercase letters,
    /// numbers, and underscores. They start and end with a letter or number.
    async fn create_namespace(
        &self,
        namespace: &NamespaceIdent,
        _properties: HashMap<String, String>,
    ) -> Result<Namespace> {
        let req = self
            .s3tables_client
            .create_namespace()
            .table_bucket_arn(self.config.table_bucket_arn.clone())
            .namespace(namespace.to_url_string());
        req.send().await.map_err(from_aws_sdk_error)?;
        Ok(Namespace::with_properties(
            namespace.clone(),
            HashMap::new(),
        ))
    }

    /// Retrieves a namespace by its identifier.
    async fn get_namespace(&self, namespace: &NamespaceIdent) -> Result<Namespace> {
        let req = self
            .s3tables_client
            .get_namespace()
            .table_bucket_arn(self.config.table_bucket_arn.clone())
            .namespace(namespace.to_url_string());
        let resp: GetNamespaceOutput = req.send().await.map_err(from_aws_sdk_error)?;
        let properties = HashMap::new();
        Ok(Namespace::with_properties(
            NamespaceIdent::from_vec(resp.namespace().to_vec())?,
            properties,
        ))
    }

    /// Checks if a namespace exists within the s3tables catalog.
    ///
    /// A service `IsNotFoundException` returns `Ok(false)`. Every other failure returns `Err`.
    async fn namespace_exists(&self, namespace: &NamespaceIdent) -> Result<bool> {
        let req = self
            .s3tables_client
            .get_namespace()
            .table_bucket_arn(self.config.table_bucket_arn.clone())
            .namespace(namespace.to_url_string());
        match req.send().await {
            Ok(_) => Ok(true),
            Err(err) => {
                if err.as_service_error().map(|e| e.is_not_found_exception()) == Some(true) {
                    Ok(false)
                } else {
                    Err(from_aws_sdk_error(err))
                }
            }
        }
    }

    /// Always fails. S3Tables does not support namespace properties.
    async fn update_namespace(
        &self,
        _namespace: &NamespaceIdent,
        _properties: HashMap<String, String>,
    ) -> Result<()> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Update namespace is not supported for s3tables catalog",
        ))
    }

    /// Drops an existing namespace from the s3tables catalog.
    async fn drop_namespace(&self, namespace: &NamespaceIdent) -> Result<()> {
        let req = self
            .s3tables_client
            .delete_namespace()
            .table_bucket_arn(self.config.table_bucket_arn.clone())
            .namespace(namespace.to_url_string());
        req.send().await.map_err(from_aws_sdk_error)?;
        Ok(())
    }

    /// Lists all tables within a given namespace.
    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        let mut result = Vec::new();
        let mut continuation_token = None;
        loop {
            let mut req = self
                .s3tables_client
                .list_tables()
                .table_bucket_arn(self.config.table_bucket_arn.clone())
                .namespace(namespace.to_url_string());
            if let Some(token) = continuation_token {
                req = req.continuation_token(token);
            }
            let resp: ListTablesOutput = req.send().await.map_err(from_aws_sdk_error)?;
            for table in resp.tables() {
                result.push(TableIdent::new(
                    NamespaceIdent::from_vec(table.namespace().to_vec())?,
                    table.name().to_string(),
                ));
            }
            continuation_token = resp.continuation_token().map(|s| s.to_string());
            if continuation_token.is_none() {
                break;
            }
        }
        Ok(result)
    }

    /// Creates a new table within a specified namespace.
    ///
    /// The s3tables catalog picks the warehouse location, so the caller must not set one.
    /// The location is only readable after the create call returns.
    async fn create_table(
        &self,
        namespace: &NamespaceIdent,
        mut creation: TableCreation,
    ) -> Result<Table> {
        let table_ident = TableIdent::new(namespace.clone(), creation.name.clone());

        let create_resp: CreateTableOutput = self
            .s3tables_client
            .create_table()
            .table_bucket_arn(self.config.table_bucket_arn.clone())
            .namespace(namespace.to_url_string())
            .format(OpenTableFormat::Iceberg)
            .name(table_ident.name())
            .send()
            .await
            .map_err(from_aws_sdk_error)?;

        // The s3tables catalog generates the warehouse location, for example
        // s3://e6c9bf20-991a-46fb-kni5xs1q2yxi3xxdyxzjzigdeop1quse2b--table-s3
        let table_location = match &creation.location {
            Some(_) => {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "The location of the table is generated by s3tables catalog, can't be set by user.",
                ));
            }
            None => {
                let get_resp: GetTableOutput = self
                    .s3tables_client
                    .get_table()
                    .table_bucket_arn(self.config.table_bucket_arn.clone())
                    .namespace(namespace.to_url_string())
                    .name(table_ident.name())
                    .send()
                    .await
                    .map_err(from_aws_sdk_error)?;
                get_resp.warehouse_location().to_string()
            }
        };

        creation.location = Some(table_location.clone());
        let metadata = TableMetadataBuilder::from_table_creation(creation)?
            .build()?
            .metadata;
        let metadata_location =
            MetadataLocation::new_with_table_location(table_location).to_string();
        metadata.write_to(&self.file_io, &metadata_location).await?;

        self.s3tables_client
            .update_table_metadata_location()
            .table_bucket_arn(self.config.table_bucket_arn.clone())
            .namespace(namespace.to_url_string())
            .name(table_ident.name())
            .metadata_location(metadata_location.clone())
            .version_token(create_resp.version_token())
            .send()
            .await
            .map_err(from_aws_sdk_error)?;

        let table = Table::builder()
            .identifier(table_ident)
            .metadata_location(metadata_location)
            .metadata(metadata)
            .file_io(self.file_io.clone())
            .build()?;
        Ok(table)
    }

    /// Loads an existing table from the s3tables catalog.
    ///
    /// A table with no metadata location fails with `Unexpected`.
    async fn load_table(&self, table_ident: &TableIdent) -> Result<Table> {
        Ok(self.load_table_with_version_token(table_ident).await?.0)
    }

    /// Drops an existing table from the s3tables catalog.
    async fn drop_table(&self, table: &TableIdent) -> Result<()> {
        let req = self
            .s3tables_client
            .delete_table()
            .table_bucket_arn(self.config.table_bucket_arn.clone())
            .namespace(table.namespace().to_url_string())
            .name(table.name());
        req.send().await.map_err(from_aws_sdk_error)?;
        Ok(())
    }

    /// Checks if a table exists within the s3tables catalog.
    ///
    /// A service `IsNotFoundException` returns `Ok(false)`. Every other failure returns `Err`.
    async fn table_exists(&self, table_ident: &TableIdent) -> Result<bool> {
        let req = self
            .s3tables_client
            .get_table()
            .table_bucket_arn(self.config.table_bucket_arn.clone())
            .namespace(table_ident.namespace().to_url_string())
            .name(table_ident.name());
        match req.send().await {
            Ok(_) => Ok(true),
            Err(err) => {
                if err.as_service_error().map(|e| e.is_not_found_exception()) == Some(true) {
                    Ok(false)
                } else {
                    Err(from_aws_sdk_error(err))
                }
            }
        }
    }

    /// Renames an existing table within the s3tables catalog.
    async fn rename_table(&self, src: &TableIdent, dest: &TableIdent) -> Result<()> {
        let req = self
            .s3tables_client
            .rename_table()
            .table_bucket_arn(self.config.table_bucket_arn.clone())
            .namespace(src.namespace().to_url_string())
            .name(src.name())
            .new_namespace_name(dest.namespace().to_url_string())
            .new_name(dest.name());
        req.send().await.map_err(from_aws_sdk_error)?;
        Ok(())
    }

    async fn register_table(
        &self,
        _table_ident: &TableIdent,
        _metadata_location: String,
    ) -> Result<Table> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Registering a table is not supported yet",
        ))
    }

    /// Updates an existing table within the s3tables catalog.
    async fn update_table(&self, mut commit: TableCommit) -> Result<Table> {
        let table_ident = commit.identifier().clone();
        let table_namespace = table_ident.namespace();
        // Skip the second full S3 metadata parse when the service pointer still matches the
        // commit base and the Transaction supplied a base table.
        let (current_table, version_token) =
            self.resolve_commit_base(&table_ident, &mut commit).await?;

        let staged_table = commit.apply(current_table)?;
        let staged_metadata_location = staged_table.metadata_location_result()?;

        staged_table
            .metadata()
            .write_to(staged_table.file_io(), staged_metadata_location)
            .await?;

        self.cas_update_metadata_location(
            &table_ident,
            table_namespace,
            version_token,
            staged_metadata_location,
        )
        .await?;

        Ok(staged_table)
    }

    /// Atomically publish a fully staged replace through a metadata-pointer CAS.
    ///
    /// `CREATE OR REPLACE TABLE ... AS SELECT` stages files under the existing table location,
    /// then calls this to swap the catalog pointer.
    ///
    /// # Errors
    ///
    /// A `Some(expected_base_metadata_location)` that does not match the service-current
    /// pointer returns a retryable [`ErrorKind::CatalogCommitConflicts`] before any update.
    async fn publish_replace_table(
        &self,
        table: Table,
        expected_base_metadata_location: Option<String>,
    ) -> Result<Table> {
        let table_ident = table.identifier().clone();
        let table_namespace = table_ident.namespace();
        // Pointer-only GetTable. The location check and the CAS never need the metadata JSON.
        let (stored, version_token) = self.get_table_pointer(&table_ident).await?;

        if let Some(expected) = expected_base_metadata_location.as_deref()
            && stored != expected
        {
            return Err(Error::new(
                ErrorKind::CatalogCommitConflicts,
                format!(
                    "Cannot publish replace for table {table_ident}: concurrent modification \
                     (expected base metadata location {expected}, found {stored})"
                ),
            )
            .with_retryable(true));
        }

        let new_metadata_location = table.metadata_location_result()?.to_string();
        // The staged replace already wrote the new metadata file. Only the pointer CAS remains.
        self.cas_update_metadata_location(
            &table_ident,
            table_namespace,
            version_token,
            &new_metadata_location,
        )
        .await?;

        Ok(table)
    }
}

impl S3TablesCatalog {
    /// CAS the table's metadata pointer via S3 Tables `UpdateTableMetadataLocation`.
    async fn cas_update_metadata_location(
        &self,
        table_ident: &TableIdent,
        table_namespace: &NamespaceIdent,
        version_token: String,
        metadata_location: &str,
    ) -> Result<()> {
        let builder = self
            .s3tables_client
            .update_table_metadata_location()
            .table_bucket_arn(&self.config.table_bucket_arn)
            .namespace(table_namespace.to_url_string())
            .name(table_ident.name())
            .version_token(version_token)
            .metadata_location(metadata_location);

        // S3 Tables maintenance commits concurrently with every writer, so an ambiguous
        // outcome here is common. A retry of an applied commit duplicates rows. So a failure
        // that may have reached the service maps to `CommitStateUnknown` (row R157).
        let _ = builder
            .send()
            .await
            .map_err(|e| match classify_commit_send_disposition(&e) {
                CommitSendDisposition::MaybeSent => Error::new(
                    ErrorKind::CommitStateUnknown,
                    format!(
                        "Commit outcome unknown for table {table_ident}: the update request \
                         may have reached S3 Tables before the failure. Verify whether the \
                         commit landed before retrying: retrying an already-applied commit \
                         duplicates its changes."
                    ),
                )
                .with_source(anyhow::Error::msg(format!("aws sdk error: {e:?}"))),
                CommitSendDisposition::NeverSent => Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "Operation failed for table: {table_ident} before the update request \
                         was sent"
                    ),
                )
                .with_source(anyhow::Error::msg(format!("aws sdk error: {e:?}"))),
                CommitSendDisposition::ResponseReceived => {
                    map_update_table_metadata_location_service_error(
                        e.into_service_error(),
                        table_ident,
                    )
                }
            })?;
        Ok(())
    }
}

/// Format AWS SDK error into iceberg error
pub(crate) fn from_aws_sdk_error<T>(error: aws_sdk_s3tables::error::SdkError<T>) -> Error
where T: std::fmt::Debug {
    Error::new(
        ErrorKind::Unexpected,
        format!("Operation failed for hitting aws sdk error: {error:?}"),
    )
}

/// Where a failed AWS SDK commit call stopped, classified sent-vs-unsent (row R157).
///
/// The Glue catalog holds a copy of this classifier. The two AWS SDK crates share no common
/// crate that can host one copy.
enum CommitSendDisposition {
    /// The request never left the client. The failure keeps its terminal mapping.
    NeverSent,
    /// The request may have reached the service. The commit outcome is ambiguous.
    MaybeSent,
    /// The service definitively responded with a modeled error.
    ResponseReceived,
}

/// Classify the transport layer of a failed SDK call on the commit path.
///
/// Java `BaseMetastoreTableOperations.checkCommitStatus` reports `UNKNOWN` for the same class
/// of failure. The SDK cannot tell connect-refused from reset-after-send, so this function
/// picks the ambiguous side. A needless reconciliation is safe. A duplicate commit is not.
fn classify_commit_send_disposition<E, R>(
    error: &aws_sdk_s3tables::error::SdkError<E, R>,
) -> CommitSendDisposition {
    use aws_sdk_s3tables::error::SdkError;
    match error {
        SdkError::ConstructionFailure(_) => CommitSendDisposition::NeverSent,
        SdkError::DispatchFailure(dispatch) if dispatch.is_user() || dispatch.is_other() => {
            CommitSendDisposition::NeverSent
        }
        SdkError::ServiceError(_) => CommitSendDisposition::ResponseReceived,
        _ => CommitSendDisposition::MaybeSent,
    }
}

/// Map a modeled `UpdateTableMetadataLocationError` on the commit path.
///
/// `ConflictException` is the version-token CAS conflict, so it stays retryable.
/// `InternalServerErrorException` may have applied the update, so it maps to
/// `CommitStateUnknown`. Java `ErrorHandlers$CommitErrorHandler` maps 500 the same way.
fn map_update_table_metadata_location_service_error(
    error: UpdateTableMetadataLocationError,
    table_ident: &TableIdent,
) -> Error {
    match error {
        UpdateTableMetadataLocationError::ConflictException(_) => Error::new(
            ErrorKind::CatalogCommitConflicts,
            format!("Commit conflicted for table: {table_ident}"),
        )
        .with_retryable(true),
        UpdateTableMetadataLocationError::NotFoundException(_) => Error::new(
            ErrorKind::TableNotFound,
            format!("Table {table_ident} is not found"),
        ),
        UpdateTableMetadataLocationError::InternalServerErrorException(_) => Error::new(
            ErrorKind::CommitStateUnknown,
            format!(
                "Commit outcome unknown for table {table_ident}: S3 Tables failed while \
                 processing the update — it may have been applied. Verify before retrying: \
                 retrying an already-applied commit duplicates its changes."
            ),
        ),
        _ => Error::new(
            ErrorKind::Unexpected,
            "Operation failed for hitting aws sdk error",
        ),
    }
    .with_source(anyhow::Error::msg(format!("aws sdk error: {error:?}")))
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    use iceberg::transaction::{ApplyTransactionAction, Transaction};

    use super::*;

    fn test_table_ident() -> TableIdent {
        TableIdent::from_strs(["ns1", "test1"]).expect("build test table ident")
    }

    /// Risk: a reclassified `ConflictException` stops the retry loop from absorbing routine
    /// concurrency, so every maintenance race reaches the caller. Pins the conflict as
    /// `CatalogCommitConflicts` and retryable.
    #[test]
    fn test_conflict_exception_stays_retryable_conflict() {
        let error = map_update_table_metadata_location_service_error(
            UpdateTableMetadataLocationError::ConflictException(
                aws_sdk_s3tables::types::error::ConflictException::builder().build(),
            ),
            &test_table_ident(),
        );
        assert_eq!(error.kind(), iceberg::ErrorKind::CatalogCommitConflicts);
        assert!(error.retryable(), "a CAS conflict is safely retryable");
    }

    /// Risk: a 5xx that keeps a terminal mapping hides may-have-landed from the caller. Pins
    /// the unknown-outcome mapping as non-retryable. Also pins that `NotFoundException` stays
    /// terminal, which is the over-broadening direction.
    #[test]
    fn test_internal_server_error_maps_to_unknown_outcome_but_not_found_stays_terminal() {
        let unknown = map_update_table_metadata_location_service_error(
            UpdateTableMetadataLocationError::InternalServerErrorException(
                aws_sdk_s3tables::types::error::InternalServerErrorException::builder().build(),
            ),
            &test_table_ident(),
        );
        assert_eq!(unknown.kind(), iceberg::ErrorKind::CommitStateUnknown);
        assert!(
            !unknown.retryable(),
            "an unknown-outcome commit error must not advertise retryability"
        );

        let not_found = map_update_table_metadata_location_service_error(
            UpdateTableMetadataLocationError::NotFoundException(
                aws_sdk_s3tables::types::error::NotFoundException::builder().build(),
            ),
            &test_table_ident(),
        );
        assert_eq!(not_found.kind(), iceberg::ErrorKind::TableNotFound);
    }

    /// Risk: a post-send ambiguous failure classifies NeverSent, so an outer re-run duplicates
    /// an applied commit. The opposite misroute costs a needless reconciliation. Pins both
    /// sides of the classifier.
    #[test]
    fn test_commit_send_disposition_split() {
        use aws_sdk_s3tables::error::ConnectorError;
        type TestSdkError = aws_sdk_s3tables::error::SdkError<(), ()>;
        fn boxed(msg: &str) -> Box<dyn std::error::Error + Send + Sync> {
            msg.to_string().into()
        }

        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::timeout_error(boxed("timed out"))),
            CommitSendDisposition::MaybeSent
        ));
        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::dispatch_failure(ConnectorError::io(
                boxed("reset mid-exchange")
            ))),
            CommitSendDisposition::MaybeSent
        ));
        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::response_error(
                boxed("unparsable response"),
                ()
            )),
            CommitSendDisposition::MaybeSent
        ));
        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::construction_failure(boxed(
                "invalid request"
            ))),
            CommitSendDisposition::NeverSent
        ));
        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::dispatch_failure(
                ConnectorError::user(boxed("client-side setup failure"))
            )),
            CommitSendDisposition::NeverSent
        ));
        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::service_error((), ())),
            CommitSendDisposition::ResponseReceived
        ));
    }

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

    /// Risk: the raw prop map holds live AWS credentials, so a derived `Debug` prints them.
    /// Pins that secret values redact to `"***"` and that keys stay visible.
    /// Mutation: revert the manual `Debug` to `#[derive(Debug)]` gives RED.
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

    /// Risk: `S3TablesCatalog` derives `Debug`, so the redaction must survive one level up.
    /// Pins that a `{:?}` of the whole catalog leaks no credential.
    /// Mutation: revert the config `Debug` to derived gives RED.
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

    /// Construction builds the SDK client but makes no network call, so these accessors need
    /// no credentials and no live bucket.
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
        // Mutation guard: the empty-map default fails this.
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
    async fn test_invalidate_defaults_are_noops() {
        let config = S3TablesCatalogConfig {
            name: Some("s3t_cat".to_string()),
            table_bucket_arn: "arn:aws:s3tables:us-east-1:123456789012:bucket/example".to_string(),
            endpoint_url: None,
            client: None,
            props: HashMap::new(),
        };
        let catalog = S3TablesCatalog::new(config, None).await.unwrap();
        let ident = TableIdent::new(NamespaceIdent::new("ns".to_string()), "t".to_string());
        // No network: the inherited no-op defaults return Ok.
        catalog.invalidate_table(&ident).await.unwrap();
        catalog.invalidate_view(&ident).await.unwrap();
    }
}
