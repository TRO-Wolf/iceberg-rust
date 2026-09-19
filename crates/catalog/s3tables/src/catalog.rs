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
use aws_sdk_s3tables::types::OpenTableFormat;
use iceberg::io::object_cache::ObjectCache;
use iceberg::io::{FileIO, FileIOBuilder, StorageFactory};
use iceberg::spec::TableMetadataBuilder;
use iceberg::table::Table;
use iceberg::{
    CacheScope, Catalog, CatalogBuilder, CommitBaseLoadPlan, Error, ErrorKind, MetadataLocation,
    Namespace, NamespaceIdent, Result, TableCommit, TableCreation, TableIdent, TableMetadataCache,
    UNNAMED_CATALOG, commit_base_conflict_error, load_or_fetch_table_metadata,
    plan_commit_base_load,
};
use iceberg_storage_opendal::OpenDalStorageFactory;

#[cfg(test)]
use crate::commit_transport::S3TablesCommitHarness;
#[cfg(test)]
use crate::commit_transport::s3tables_commit_send_landed;
use crate::commit_transport::{
    LiveS3TablesCommitTransport, S3TablesCommitSend, S3TablesCommitTransport, S3TablesUpdateCall,
    map_s3tables_commit_send,
};
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
    pub(crate) table_metadata_cache: Option<Arc<TableMetadataCache>>,
    pub(crate) shared_object_cache_bytes: Option<u64>,
    pub(crate) cache_credential_context: Option<String>,
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
            table_metadata_cache: None,
            shared_object_cache_bytes: None,
            cache_credential_context: None,
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
                S3TablesCatalog::new(self.config, self.storage_factory)
                    .await
                    .map(|catalog| {
                        catalog.with_cache_options(
                            self.table_metadata_cache,
                            self.shared_object_cache_bytes,
                            self.cache_credential_context,
                        )
                    })
            }
        }
    }
}

/// S3Tables catalog implementation.
pub struct S3TablesCatalog {
    config: S3TablesCatalogConfig,
    s3tables_client: aws_sdk_s3tables::Client,
    pub(crate) file_io: FileIO,
    commit_transport: Arc<dyn S3TablesCommitTransport>,
    pub(crate) table_metadata_cache: Option<Arc<TableMetadataCache>>,
    pub(crate) cache_scope: CacheScope,
    pub(crate) shared_object_cache: Option<Arc<ObjectCache>>,
    #[cfg(test)]
    pub(crate) pointer_source: Option<PointerSource>,
    #[cfg(test)]
    drop_source: Option<DropSource>,
    #[cfg(test)]
    outcome_harness: Option<Arc<S3TablesCommitHarness>>,
}

#[cfg(test)]
pub(crate) type PointerSource = Arc<dyn Fn(&TableIdent) -> Result<(String, String)> + Send + Sync>;
#[cfg(test)]
pub(crate) type DropSource = Arc<dyn Fn(&TableIdent) -> Result<()> + Send + Sync>;

impl std::fmt::Debug for S3TablesCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3TablesCatalog")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
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

        let injected_io = config.client.is_some() || storage_factory.is_some();

        let factory = storage_factory.unwrap_or_else(|| {
            Arc::new(OpenDalStorageFactory::S3 {
                configured_scheme: "s3".to_string(),
                customized_credential_load: None,
            })
        });
        let file_io = FileIOBuilder::new(factory)
            .with_props(&config.props)
            .build();

        let commit_transport = Arc::new(LiveS3TablesCommitTransport::new(s3tables_client.clone()));

        let cache_scope = if injected_io {
            CacheScope::isolated(format!("s3tables:{}", config.table_bucket_arn))
        } else {
            CacheScope::for_catalog(
                format!("s3tables:{}", config.table_bucket_arn),
                None,
                &config.props,
            )
        };

        Ok(Self {
            config,
            s3tables_client,
            file_io,
            commit_transport,
            table_metadata_cache: None,
            cache_scope,
            shared_object_cache: None,
            #[cfg(test)]
            pointer_source: None,
            #[cfg(test)]
            drop_source: None,
            #[cfg(test)]
            outcome_harness: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_pointer_source(mut self, source: PointerSource) -> Self {
        self.pointer_source = Some(source);
        self
    }

    #[cfg(test)]
    pub(crate) fn with_drop_source(mut self, source: DropSource) -> Self {
        self.drop_source = Some(source);
        self
    }

    #[cfg(test)]
    pub(crate) fn with_file_io_for_tests(mut self, file_io: FileIO) -> Self {
        self.file_io = file_io;
        self
    }

    #[cfg(test)]
    pub(crate) fn catalog_commit_attempts(&self) -> u64 {
        self.commit_transport.catalog_commit_attempts()
    }

    #[cfg(test)]
    pub(crate) fn with_commit_transport(
        mut self,
        commit_transport: Arc<dyn S3TablesCommitTransport>,
    ) -> Self {
        self.commit_transport = commit_transport;
        self
    }

    #[cfg(test)]
    pub(crate) fn live_commit_transport(&self) -> Arc<dyn S3TablesCommitTransport> {
        Arc::clone(&self.commit_transport)
    }

    #[cfg(test)]
    pub(crate) fn for_commit_outcome_tests(
        file_io: FileIO,
        commit_transport: Arc<dyn S3TablesCommitTransport>,
        table: Table,
        client: aws_sdk_s3tables::Client,
    ) -> Self {
        let harness = S3TablesCommitHarness::new(table, "v0".to_string());
        Self {
            config: S3TablesCatalogConfig {
                name: Some("pr5a-s3tables".to_string()),
                table_bucket_arn: "arn:aws:s3tables:us-east-1:123456789012:bucket/pr5a".to_string(),
                endpoint_url: None,
                client: None,
                props: HashMap::new(),
            },
            s3tables_client: client,
            file_io,
            commit_transport,
            table_metadata_cache: None,
            cache_scope: CacheScope::isolated("s3tables:test"),
            shared_object_cache: None,
            pointer_source: None,
            drop_source: None,
            outcome_harness: Some(harness),
        }
    }

    /// GetTable for the service metadata pointer and version_token. Reads no object storage.
    async fn get_table_pointer(
        &self,
        table_ident: &TableIdent,
    ) -> Result<(
        String, /* metadata_location */
        String, /* version_token */
    )> {
        #[cfg(test)]
        if let Some(source) = &self.pointer_source {
            return source(table_ident);
        }
        #[cfg(test)]
        if let Some(harness) = &self.outcome_harness {
            return Ok(harness.pointer());
        }
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
        #[cfg(test)]
        if let Some(harness) = &self.outcome_harness {
            let loaded = harness.table();
            let version_token = harness.pointer().1;
            let rebound = self
                .table_builder()
                .identifier(table_ident.clone())
                .metadata(loaded.metadata_ref())
                .metadata_location(
                    loaded
                        .metadata_location()
                        .ok_or_else(|| {
                            Error::new(
                                ErrorKind::Unexpected,
                                format!(
                                    "Table {} does not have metadata location",
                                    table_ident.name()
                                ),
                            )
                        })?
                        .to_string(),
                )
                .build()?;
            return Ok((rebound, version_token));
        }
        let (metadata_location, version_token) = self.get_table_pointer(table_ident).await?;
        let metadata = load_or_fetch_table_metadata(
            &self.file_io,
            &self.cache_scope,
            &metadata_location,
            self.table_metadata_cache.as_deref(),
            Some(&version_token),
        )
        .await?;

        let table = self
            .table_builder()
            .identifier(table_ident.clone())
            .metadata(metadata)
            .metadata_location(metadata_location)
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
                let table = self
                    .table_builder()
                    .identifier(table_ident.clone())
                    .metadata(provided.metadata_ref())
                    .metadata_location(service_location)
                    .build()?;
                Ok((table, version_token))
            }
            CommitBaseLoadPlan::Conflict => Err(commit_base_conflict_error(
                table_ident,
                base_loc.as_deref(),
                &service_location,
            )),
            CommitBaseLoadPlan::FullLoad => {
                let metadata = load_or_fetch_table_metadata(
                    &self.file_io,
                    &self.cache_scope,
                    &service_location,
                    self.table_metadata_cache.as_deref(),
                    Some(&version_token),
                )
                .await?;
                let table = self
                    .table_builder()
                    .identifier(table_ident.clone())
                    .metadata(metadata)
                    .metadata_location(service_location)
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

        let update_resp = self
            .s3tables_client
            .update_table_metadata_location()
            .table_bucket_arn(self.config.table_bucket_arn.clone())
            .namespace(namespace.to_url_string())
            .name(table_ident.name())
            .metadata_location(metadata_location.clone())
            .version_token(create_resp.version_token())
            .send()
            .await
            .map_err(from_aws_sdk_error)?;

        let table = self
            .table_builder()
            .identifier(table_ident)
            .metadata_location(metadata_location.clone())
            .metadata(metadata)
            .build()?;

        self.cache_put(
            &metadata_location,
            table.metadata_ref(),
            Some(update_resp.version_token),
        )
        .await;

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
        let dropped_location = self
            .get_table_pointer(table)
            .await
            .ok()
            .map(|(location, _)| location);

        #[cfg(test)]
        if let Some(source) = &self.drop_source {
            source(table)?;
        }
        #[cfg(test)]
        if self.drop_source.is_none() {
            let req = self
                .s3tables_client
                .delete_table()
                .table_bucket_arn(self.config.table_bucket_arn.clone())
                .namespace(table.namespace().to_url_string())
                .name(table.name());
            req.send().await.map_err(from_aws_sdk_error)?;
        }
        #[cfg(not(test))]
        {
            let req = self
                .s3tables_client
                .delete_table()
                .table_bucket_arn(self.config.table_bucket_arn.clone())
                .namespace(table.namespace().to_url_string())
                .name(table.name());
            req.send().await.map_err(from_aws_sdk_error)?;
        }

        if let (Some(cache), Some(location)) = (
            self.table_metadata_cache.as_ref(),
            dropped_location.as_deref(),
        ) {
            cache.invalidate(&self.cache_scope, location).await;
        }
        Ok(())
    }

    async fn invalidate_table(&self, table: &TableIdent) -> Result<()> {
        let Some(cache) = self.table_metadata_cache.as_ref() else {
            return Ok(());
        };
        let (location, _) = self.get_table_pointer(table).await?;
        cache.invalidate(&self.cache_scope, &location).await;
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
            "S3 Tables has no register-by-metadata-location operation: the Iceberg REST register endpoint is not in the service mapping, and UpdateTableMetadataLocation requires a URI under the table warehouse (GAP_MATRIX row R126)",
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
            .write_commit_metadata(staged_table.file_io(), staged_metadata_location)
            .await?;

        let new_version = self
            .cas_update_metadata_location(
                &table_ident,
                table_namespace,
                version_token,
                staged_metadata_location,
                Some(&staged_table),
            )
            .await?;

        self.cache_put(
            staged_metadata_location,
            staged_table.metadata_ref(),
            new_version,
        )
        .await;

        Ok(staged_table)
    }

    /// Atomically publish a fully staged replace through a metadata-pointer CAS. `CREATE OR REPLACE
    /// TABLE ... AS SELECT` stages files under the existing table location, then calls this to swap
    /// the catalog pointer. # Errors A `Some(expected_base_metadata_location)` that does not match
    /// the service-current pointer returns a retryable [`ErrorKind::CatalogCommitConflicts`] before
    /// any update.
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
        let new_version = self
            .cas_update_metadata_location(
                &table_ident,
                table_namespace,
                version_token,
                &new_metadata_location,
                Some(&table),
            )
            .await?;

        self.cache_put(&new_metadata_location, table.metadata_ref(), new_version)
            .await;

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
        #[cfg_attr(not(test), allow(unused_variables))] published: Option<&Table>,
    ) -> Result<Option<String>> {
        let send = self
            .commit_transport
            .send_update_metadata_location(S3TablesUpdateCall {
                table_bucket_arn: self.config.table_bucket_arn.clone(),
                namespace: table_namespace.to_url_string(),
                name: table_ident.name().to_string(),
                version_token,
                metadata_location: metadata_location.to_string(),
            })
            .await;
        #[cfg(test)]
        if s3tables_commit_send_landed(&send)
            && let Some(table) = published
        {
            self.publish_outcome_harness(table);
        }
        let new_version = match &send {
            S3TablesCommitSend::Success(token) => token.clone(),
            _ => None,
        };
        map_s3tables_commit_send(send, table_ident)?;
        Ok(new_version)
    }

    #[cfg(test)]
    fn publish_outcome_harness(&self, table: &Table) {
        if let Some(harness) = &self.outcome_harness {
            harness.publish(table.clone());
        }
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

#[cfg(test)]
mod tests {
    include!("catalog_tests.rs");
}

#[cfg(test)]
mod cache_tests {
    include!("cache_tests.rs");
}
