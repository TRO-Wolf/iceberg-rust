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

//! This module contains the iceberg REST catalog implementation.

use std::collections::HashMap;
use std::future::Future;
use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use iceberg::io::{FileIO, FileIOBuilder, StorageFactory};
use iceberg::table::Table;
use iceberg::view::{View, ViewCommit};
use iceberg::{
    Catalog, CatalogBuilder, Error, ErrorKind, Namespace, NamespaceIdent, Result, TableCommit,
    TableCreation, TableIdent, UNNAMED_CATALOG, ViewCreation,
};
use itertools::Itertools;
use reqwest::header::{
    HeaderMap, HeaderName, HeaderValue, {self},
};
use reqwest::{Client, Method, StatusCode, Url};
use tokio::sync::OnceCell;
use typed_builder::TypedBuilder;

use crate::client::{
    HttpClient, deserialize_catalog_response, deserialize_unexpected_catalog_error,
};
use crate::types::{
    CatalogConfig, CommitTableRequest, CommitTableResponse, CommitViewRequest,
    CreateNamespaceRequest, CreateTableRequest, CreateViewRequest, ListNamespaceResponse,
    ListTablesResponse, LoadTableResult, LoadViewResult, NamespaceResponse, RegisterTableRequest,
    RenameTableRequest, StorageCredential,
};

/// REST catalog URI — the base address of the Iceberg REST catalog service.
///
/// # Notes
///
/// The `/v1/config` response can replace this value and `oauth2-server-uri`; server overrides
/// outrank operator properties, as in Java `RESTSessionCatalog.initialize`. The config fetch uses
/// pre-merge properties, so the response cannot redirect the request that fetched it. Like Java,
/// the client applies no host, IP-range, or scheme filter: a private-IP blocklist would break
/// private-endpoint deployments. Validate an untrusted catalog URI before it reaches this key.
pub const REST_CATALOG_PROP_URI: &str = "uri";
/// REST catalog warehouse location
pub const REST_CATALOG_PROP_WAREHOUSE: &str = "warehouse";
/// Disable header redaction in error logs (defaults to false for security)
pub const REST_CATALOG_PROP_DISABLE_HEADER_REDACTION: &str = "disable-header-redaction";

const ICEBERG_REST_SPEC_VERSION: &str = "0.14.1";
const CARGO_PKG_VERSION: &str = env!("CARGO_PKG_VERSION");
const PATH_V1: &str = "v1";

/// Select the vended storage credential whose `prefix` most specifically covers `storage_path`.
/// Mirrors Java `S3FileIO.clientForStoragePath`.
///
/// The longest matching prefix wins. On a length tie the fork keeps the first in list order;
/// Java iterates a `HashMap` and picks an arbitrary one. `None` means no credential matches,
/// which the caller must treat as Java does: fall back to the un-vended client, raise no error.
fn select_vended_credential<'a>(
    storage_path: Option<&str>,
    storage_credentials: Option<&'a [StorageCredential]>,
) -> Option<&'a StorageCredential> {
    let path = storage_path?;
    let mut best: Option<&StorageCredential> = None;
    for candidate in storage_credentials.unwrap_or(&[]) {
        if !path.starts_with(&candidate.prefix) {
            continue;
        }
        let is_longer = match best {
            Some(current) => candidate.prefix.len() > current.prefix.len(),
            None => true,
        };
        if is_longer {
            best = Some(candidate);
        }
    }
    best
}

/// Builder for [`RestCatalog`].
#[derive(Debug)]
pub struct RestCatalogBuilder {
    config: RestCatalogConfig,
    storage_factory: Option<Arc<dyn StorageFactory>>,
}

impl Default for RestCatalogBuilder {
    fn default() -> Self {
        Self {
            config: RestCatalogConfig {
                name: None,
                uri: "".to_string(),
                warehouse: None,
                props: HashMap::new(),
                client: None,
            },
            storage_factory: None,
        }
    }
}

impl CatalogBuilder for RestCatalogBuilder {
    type C = RestCatalog;

    fn with_storage_factory(mut self, storage_factory: Arc<dyn StorageFactory>) -> Self {
        self.storage_factory = Some(storage_factory);
        self
    }

    fn load(
        mut self,
        name: impl Into<String>,
        props: HashMap<String, String>,
    ) -> impl Future<Output = Result<Self::C>> + Send {
        self.config.name = Some(name.into());

        if props.contains_key(REST_CATALOG_PROP_URI) {
            self.config.uri = props
                .get(REST_CATALOG_PROP_URI)
                .cloned()
                .unwrap_or_default();
        }

        if props.contains_key(REST_CATALOG_PROP_WAREHOUSE) {
            self.config.warehouse = props.get(REST_CATALOG_PROP_WAREHOUSE).cloned()
        }

        // Collect other remaining properties
        self.config.props = props
            .into_iter()
            .filter(|(k, _)| k != REST_CATALOG_PROP_URI && k != REST_CATALOG_PROP_WAREHOUSE)
            .collect();

        let result = {
            if self.config.name.is_none() {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Catalog name is required",
                ))
            } else if self.config.uri.is_empty() {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Catalog uri is required",
                ))
            } else {
                Ok(RestCatalog::new(self.config, self.storage_factory))
            }
        };

        std::future::ready(result)
    }
}

impl RestCatalogBuilder {
    /// Configures the catalog with a custom HTTP client.
    pub fn with_client(mut self, client: Client) -> Self {
        self.config.client = Some(client);
        self
    }
}

/// Returns true if a property key holds a secret value that must be redacted from `Debug`.
///
/// Delegates to the canonical needle test so this crate cannot drift from the other catalogs.
/// [`RestCatalogConfig`]`::props` is cloned into the table `FileIO` props, so it carries storage
/// credentials such as `s3.secret-access-key`; a REST-local exact-match list missed them.
///
/// The needle test is a deliberate superset. It also redacts a non-secret key that contains a
/// needle, such as `token-refresh-enabled`. Over-redaction is the safe direction for a debug view.
fn is_secret_prop_key(key: &str) -> bool {
    iceberg::io::is_secret_prop_key(key)
}

/// Rest catalog configuration.
#[derive(Clone, TypedBuilder)]
pub(crate) struct RestCatalogConfig {
    #[builder(default, setter(strip_option))]
    name: Option<String>,

    uri: String,

    #[builder(default, setter(strip_option(fallback = warehouse_opt)))]
    warehouse: Option<String>,

    #[builder(default)]
    props: HashMap<String, String>,

    #[builder(default)]
    client: Option<Client>,
}

impl std::fmt::Debug for RestCatalogConfig {
    /// Hand-written so secret-bearing `props` entries print as `"***"`. A derived `Debug` would
    /// leak the OAuth credential through any struct that embeds this config.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let redacted_props: HashMap<&str, &str> = self
            .props
            .iter()
            .map(|(k, v)| {
                if is_secret_prop_key(k) {
                    (k.as_str(), "***")
                } else {
                    (k.as_str(), v.as_str())
                }
            })
            .collect();

        f.debug_struct("RestCatalogConfig")
            .field("name", &self.name)
            .field("uri", &self.uri)
            .field("warehouse", &self.warehouse)
            .field("props", &redacted_props)
            .field("client", &self.client)
            .finish()
    }
}

impl RestCatalogConfig {
    fn url_prefixed(&self, parts: &[&str]) -> String {
        [&self.uri, PATH_V1]
            .into_iter()
            .chain(self.props.get("prefix").map(|s| &**s))
            .chain(parts.iter().cloned())
            .join("/")
    }

    fn config_endpoint(&self) -> String {
        [&self.uri, PATH_V1, "config"].join("/")
    }

    /// The OAuth token endpoint: the operator-configured `oauth2-server-uri` when present,
    /// otherwise the catalog-relative `<uri>/v1/oauth/tokens`.
    ///
    /// The configured value is used verbatim, with no scheme or address check. Java
    /// `AuthConfig.fromProperties` does the same. See the trust note on [`REST_CATALOG_PROP_URI`].
    pub(crate) fn get_token_endpoint(&self) -> String {
        if let Some(oauth2_uri) = self.props.get("oauth2-server-uri") {
            oauth2_uri.to_string()
        } else {
            [&self.uri, PATH_V1, "oauth", "tokens"].join("/")
        }
    }

    fn namespaces_endpoint(&self) -> String {
        self.url_prefixed(&["namespaces"])
    }

    fn namespace_endpoint(&self, ns: &NamespaceIdent) -> String {
        self.url_prefixed(&["namespaces", &ns.to_url_string()])
    }

    fn tables_endpoint(&self, ns: &NamespaceIdent) -> String {
        self.url_prefixed(&["namespaces", &ns.to_url_string(), "tables"])
    }

    fn rename_table_endpoint(&self) -> String {
        self.url_prefixed(&["tables", "rename"])
    }

    fn register_table_endpoint(&self, ns: &NamespaceIdent) -> String {
        self.url_prefixed(&["namespaces", &ns.to_url_string(), "register"])
    }

    fn table_endpoint(&self, table: &TableIdent) -> String {
        self.url_prefixed(&[
            "namespaces",
            &table.namespace.to_url_string(),
            "tables",
            &table.name,
        ])
    }

    fn views_endpoint(&self, ns: &NamespaceIdent) -> String {
        self.url_prefixed(&["namespaces", &ns.to_url_string(), "views"])
    }

    fn rename_view_endpoint(&self) -> String {
        self.url_prefixed(&["views", "rename"])
    }

    fn view_endpoint(&self, view: &TableIdent) -> String {
        self.url_prefixed(&[
            "namespaces",
            &view.namespace.to_url_string(),
            "views",
            &view.name,
        ])
    }

    /// Get the client from the config.
    pub(crate) fn client(&self) -> Option<Client> {
        self.client.clone()
    }

    /// Get the token from the config.
    ///
    /// The client can use this token to send requests.
    pub(crate) fn token(&self) -> Option<String> {
        self.props.get("token").cloned()
    }

    /// Get the credentials used to fetch a new token.
    ///
    /// The `credential` property is either `client_secret` or `client_id:client_secret`.
    pub(crate) fn credential(&self) -> Option<(Option<String>, String)> {
        let cred = self.props.get("credential")?;

        match cred.split_once(':') {
            Some((client_id, client_secret)) => {
                Some((Some(client_id.to_string()), client_secret.to_string()))
            }
            None => Some((None, cred.to_string())),
        }
    }

    /// Get the extra headers: `content-type`, `x-client-version`, `user-agent`, and every
    /// `header.xxx` property.
    pub(crate) fn extra_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::from_iter([
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            ),
            (
                HeaderName::from_static("x-client-version"),
                HeaderValue::from_static(ICEBERG_REST_SPEC_VERSION),
            ),
            (
                header::USER_AGENT,
                HeaderValue::from_str(&format!("iceberg-rs/{CARGO_PKG_VERSION}")).unwrap(),
            ),
        ]);

        for (key, value) in self
            .props
            .iter()
            .filter_map(|(k, v)| k.strip_prefix("header.").map(|k| (k, v)))
        {
            headers.insert(
                HeaderName::from_str(key).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid header name: {key}"),
                    )
                    .with_source(e)
                })?,
                HeaderValue::from_str(value).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid header value: {value}"),
                    )
                    .with_source(e)
                })?,
            );
        }

        Ok(headers)
    }

    /// Get the optional OAuth headers from the config.
    pub(crate) fn extra_oauth_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();

        if let Some(scope) = self.props.get("scope") {
            params.insert("scope".to_string(), scope.to_string());
        } else {
            params.insert("scope".to_string(), "catalog".to_string());
        }

        let optional_params = ["audience", "resource"];
        for param_name in optional_params {
            if let Some(value) = self.props.get(param_name) {
                params.insert(param_name.to_string(), value.to_string());
            }
        }

        params
    }

    /// Whether proactive OAuth token refresh is enabled (`token-refresh-enabled`).
    ///
    /// Mirrors Java `OAuth2Properties.TOKEN_REFRESH_ENABLED`, which defaults to true. When off,
    /// the client exchanges the credential once and caches the token forever.
    pub(crate) fn token_refresh_enabled(&self) -> bool {
        // Java `PropertyUtil.propertyAsBoolean` parses only a case-insensitive "true" as true,
        // and defaults to true when the key is absent.
        self.props
            .get("token-refresh-enabled")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(true)
    }

    /// True when `disable-header-redaction` is `"true"`. Defaults to false, so headers are
    /// redacted unless the operator opts out.
    pub(crate) fn disable_header_redaction(&self) -> bool {
        self.props
            .get(REST_CATALOG_PROP_DISABLE_HEADER_REDACTION)
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    /// Merge the `RestCatalogConfig` with the a [`CatalogConfig`] (fetched from the REST server).
    pub(crate) fn merge_with_config(mut self, mut config: CatalogConfig) -> Self {
        if let Some(uri) = config.overrides.remove("uri") {
            self.uri = uri;
        }

        // `disable-header-redaction` is a client-only logging control with no wire meaning.
        // A malicious server could set it to unmask the `Authorization` header in error logs.
        // Strip it from both server maps so the user value wins. Java has no analogue, so this
        // is fork-local hardening, not a divergence. Other server overrides stay honored.
        let user_redaction = self
            .props
            .get(REST_CATALOG_PROP_DISABLE_HEADER_REDACTION)
            .cloned();
        config
            .defaults
            .remove(REST_CATALOG_PROP_DISABLE_HEADER_REDACTION);
        config
            .overrides
            .remove(REST_CATALOG_PROP_DISABLE_HEADER_REDACTION);

        let mut props = config.defaults;
        props.extend(self.props);
        props.extend(config.overrides);

        // Restore the user's own value (or its absence) as the authoritative setting,
        // overriding anything the merge may have reintroduced.
        match user_redaction {
            Some(value) => {
                props.insert(
                    REST_CATALOG_PROP_DISABLE_HEADER_REDACTION.to_string(),
                    value,
                );
            }
            None => {
                props.remove(REST_CATALOG_PROP_DISABLE_HEADER_REDACTION);
            }
        }

        self.props = props;
        self
    }
}

#[derive(Debug)]
struct RestContext {
    client: HttpClient,
    /// Runtime config is fetched from rest server and stored here.
    ///
    /// It's could be different from the user config.
    config: RestCatalogConfig,
}

/// Rest catalog implementation.
#[derive(Debug)]
pub struct RestCatalog {
    /// User config is stored as-is and never be changed.
    ///
    /// It could be different from the config fetched from the server and used at runtime.
    user_config: RestCatalogConfig,
    ctx: OnceCell<RestContext>,
    /// Storage factory for creating FileIO instances.
    storage_factory: Option<Arc<dyn StorageFactory>>,
}

impl RestCatalog {
    /// Creates a `RestCatalog` from a [`RestCatalogConfig`].
    fn new(config: RestCatalogConfig, storage_factory: Option<Arc<dyn StorageFactory>>) -> Self {
        Self {
            user_config: config,
            ctx: OnceCell::new(),
            storage_factory,
        }
    }

    /// Gets the [`RestContext`] from the catalog.
    async fn context(&self) -> Result<&RestContext> {
        self.ctx
            .get_or_try_init(|| async {
                let client = HttpClient::new(&self.user_config)?;
                let catalog_config = RestCatalog::load_config(&client, &self.user_config).await?;
                let config = self.user_config.clone().merge_with_config(catalog_config);
                let client = client.update_with(&config)?;

                Ok(RestContext { config, client })
            })
            .await
    }

    /// Load the runtime config from the server by `user_config`.
    ///
    /// It's required for a REST catalog to update its config after creation.
    async fn load_config(
        client: &HttpClient,
        user_config: &RestCatalogConfig,
    ) -> Result<CatalogConfig> {
        let mut request_builder = client.request(Method::GET, user_config.config_endpoint());

        if let Some(warehouse_location) = &user_config.warehouse {
            request_builder = request_builder.query(&[("warehouse", warehouse_location)]);
        }

        let request = request_builder.build()?;

        let http_response = client.query_catalog(request).await?;

        match http_response.status() {
            StatusCode::OK => deserialize_catalog_response(http_response).await,
            _ => Err(deserialize_unexpected_catalog_error(
                http_response,
                client.disable_header_redaction(),
            )
            .await),
        }
    }

    async fn load_file_io(
        &self,
        metadata_location: Option<&str>,
        extra_config: Option<HashMap<String, String>>,
        storage_credentials: Option<&[StorageCredential]>,
    ) -> Result<FileIO> {
        let mut props = self.context().await?.config.props.clone();
        if let Some(config) = extra_config {
            props.extend(config);
        }

        // Overlay the vended storage credential for this table's location. Mirrors Java
        // `RESTSessionCatalog.newFileIO`. The credential config is layered LAST, so it beats the
        // catalog and table props on a key collision, as Java's `buildKeepingLast` does.
        //
        // Java re-selects the credential per accessed path in `S3FileIO.clientForStoragePath`.
        // The flat-props `FileIO` here cannot, so the fork selects once at `metadata_location`.
        // A table whose data and metadata buckets differ diverges; see GAP_MATRIX row R160.
        if let Some(credential) = select_vended_credential(metadata_location, storage_credentials) {
            props.extend(credential.config.clone());
        }

        // If the warehouse is a logical identifier instead of a URL we don't want
        // to raise an exception
        let warehouse_path = match self.context().await?.config.warehouse.as_deref() {
            Some(url) if Url::parse(url).is_ok() => Some(url),
            Some(_) => None,
            None => None,
        };

        if metadata_location.or(warehouse_path).is_none() {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Unable to load file io, neither warehouse nor metadata location is set!",
            ));
        }

        // Require a StorageFactory to be provided
        let factory = self
            .storage_factory
            .clone()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "StorageFactory must be provided for RestCatalog. Use `with_storage_factory` to configure it.",
                )
            })?;

        let file_io = FileIOBuilder::new(factory).with_props(props).build();

        Ok(file_io)
    }

    /// Build a [`View`] from a `LoadViewResult` returned by the view load/create/commit endpoints.
    ///
    /// The local `user_config` wins over the response `config` on a key collision, as the table
    /// load path does. The [`FileIO`] loads against the view's metadata location.
    async fn build_view_from_load_result(
        &self,
        view_ident: TableIdent,
        response: LoadViewResult,
    ) -> Result<View> {
        let config = response
            .config
            .into_iter()
            .chain(self.user_config.props.clone())
            .collect();

        let file_io = self
            .load_file_io(Some(&response.metadata_location), Some(config), None)
            .await?;

        View::builder()
            .identifier(view_ident)
            .file_io(file_io)
            .metadata(response.metadata)
            .metadata_location(response.metadata_location)
            .build()
    }

    /// Invalidate the current token without generating a new one. On the next request, the client
    /// will attempt to generate a new token.
    pub async fn invalidate_token(&self) -> Result<()> {
        self.context().await?.client.invalidate_token().await
    }

    /// Invalidate the current token and set a new one. The new token is fetched before the lock
    /// is taken, so callers keep using the old token until the swap.
    ///
    /// # Errors
    ///
    /// A bad credential or a failed request leaves the current token unchanged.
    pub async fn regenerate_token(&self) -> Result<()> {
        self.context().await?.client.regenerate_token().await
    }
}

/// All requests and expected responses are derived from the REST catalog API spec:
/// https://github.com/apache/iceberg/blob/main/open-api/rest-catalog-open-api.yaml
#[async_trait]
impl Catalog for RestCatalog {
    /// Returns the name given to [`CatalogBuilder::load`], or [`UNNAMED_CATALOG`]. Mirrors Java
    /// `RESTCatalog.name`.
    fn name(&self) -> &str {
        self.user_config.name.as_deref().unwrap_or(UNNAMED_CATALOG)
    }

    /// Returns the user-supplied configuration properties. Mirrors Java `RESTCatalog.properties`.
    /// These are the as-loaded props, not the server-merged runtime config.
    fn properties(&self) -> &HashMap<String, String> {
        &self.user_config.props
    }

    async fn list_namespaces(
        &self,
        parent: Option<&NamespaceIdent>,
    ) -> Result<Vec<NamespaceIdent>> {
        let context = self.context().await?;
        let endpoint = context.config.namespaces_endpoint();
        let mut namespaces = Vec::new();
        let mut next_token = None;

        loop {
            let mut request = context.client.request(Method::GET, endpoint.clone());

            // Filter on `parent={namespace}` if a parent namespace exists.
            if let Some(ns) = parent {
                request = request.query(&[("parent", ns.to_url_string())]);
            }

            if let Some(token) = next_token {
                request = request.query(&[("pageToken", token)]);
            }

            let http_response = context.client.query_catalog(request.build()?).await?;

            match http_response.status() {
                StatusCode::OK => {
                    let response =
                        deserialize_catalog_response::<ListNamespaceResponse>(http_response)
                            .await?;

                    namespaces.extend(response.namespaces);

                    match response.next_page_token {
                        Some(token) => next_token = Some(token),
                        None => break,
                    }
                }
                StatusCode::NOT_FOUND => {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        "The parent parameter of the namespace provided does not exist",
                    ));
                }
                _ => {
                    return Err(deserialize_unexpected_catalog_error(
                        http_response,
                        context.client.disable_header_redaction(),
                    )
                    .await);
                }
            }
        }

        Ok(namespaces)
    }

    async fn create_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<Namespace> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::POST, context.config.namespaces_endpoint())
            .json(&CreateNamespaceRequest {
                namespace: namespace.clone(),
                properties,
            })
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        match http_response.status() {
            StatusCode::OK => {
                let response =
                    deserialize_catalog_response::<NamespaceResponse>(http_response).await?;
                Ok(Namespace::from(response))
            }
            StatusCode::CONFLICT => Err(Error::new(
                ErrorKind::Unexpected,
                "Tried to create a namespace that already exists",
            )),
            _ => Err(deserialize_unexpected_catalog_error(
                http_response,
                context.client.disable_header_redaction(),
            )
            .await),
        }
    }

    async fn get_namespace(&self, namespace: &NamespaceIdent) -> Result<Namespace> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::GET, context.config.namespace_endpoint(namespace))
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        match http_response.status() {
            StatusCode::OK => {
                let response =
                    deserialize_catalog_response::<NamespaceResponse>(http_response).await?;
                Ok(Namespace::from(response))
            }
            StatusCode::NOT_FOUND => Err(Error::new(
                ErrorKind::Unexpected,
                "Tried to get a namespace that does not exist",
            )),
            _ => Err(deserialize_unexpected_catalog_error(
                http_response,
                context.client.disable_header_redaction(),
            )
            .await),
        }
    }

    async fn namespace_exists(&self, ns: &NamespaceIdent) -> Result<bool> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::HEAD, context.config.namespace_endpoint(ns))
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        match http_response.status() {
            StatusCode::NO_CONTENT | StatusCode::OK => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            _ => Err(deserialize_unexpected_catalog_error(
                http_response,
                context.client.disable_header_redaction(),
            )
            .await),
        }
    }

    async fn update_namespace(
        &self,
        _namespace: &NamespaceIdent,
        _properties: HashMap<String, String>,
    ) -> Result<()> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Updating namespace not supported yet!",
        ))
    }

    async fn drop_namespace(&self, namespace: &NamespaceIdent) -> Result<()> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::DELETE, context.config.namespace_endpoint(namespace))
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        match http_response.status() {
            StatusCode::NO_CONTENT | StatusCode::OK => Ok(()),
            StatusCode::NOT_FOUND => Err(Error::new(
                ErrorKind::Unexpected,
                "Tried to drop a namespace that does not exist",
            )),
            _ => Err(deserialize_unexpected_catalog_error(
                http_response,
                context.client.disable_header_redaction(),
            )
            .await),
        }
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        let context = self.context().await?;
        let endpoint = context.config.tables_endpoint(namespace);
        let mut identifiers = Vec::new();
        let mut next_token = None;

        loop {
            let mut request = context.client.request(Method::GET, endpoint.clone());

            if let Some(token) = next_token {
                request = request.query(&[("pageToken", token)]);
            }

            let http_response = context.client.query_catalog(request.build()?).await?;

            match http_response.status() {
                StatusCode::OK => {
                    let response =
                        deserialize_catalog_response::<ListTablesResponse>(http_response).await?;

                    identifiers.extend(response.identifiers);

                    match response.next_page_token {
                        Some(token) => next_token = Some(token),
                        None => break,
                    }
                }
                StatusCode::NOT_FOUND => {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        "Tried to list tables of a namespace that does not exist",
                    ));
                }
                _ => {
                    return Err(deserialize_unexpected_catalog_error(
                        http_response,
                        context.client.disable_header_redaction(),
                    )
                    .await);
                }
            }
        }

        Ok(identifiers)
    }

    /// Create a new table inside the namespace.
    ///
    /// The local `RestCatalog` config wins over the server response on a property collision.
    async fn create_table(
        &self,
        namespace: &NamespaceIdent,
        creation: TableCreation,
    ) -> Result<Table> {
        let context = self.context().await?;

        let table_ident = TableIdent::new(namespace.clone(), creation.name.clone());

        let request = context
            .client
            .request(Method::POST, context.config.tables_endpoint(namespace))
            .json(&CreateTableRequest {
                name: creation.name,
                location: creation.location,
                schema: creation.schema,
                partition_spec: creation.partition_spec,
                write_order: creation.sort_order,
                stage_create: Some(false),
                properties: creation.properties,
            })
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        let response = match http_response.status() {
            StatusCode::OK => {
                deserialize_catalog_response::<LoadTableResult>(http_response).await?
            }
            StatusCode::NOT_FOUND => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    "Tried to create a table under a namespace that does not exist",
                ));
            }
            StatusCode::CONFLICT => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    "The table already exists",
                ));
            }
            _ => {
                return Err(deserialize_unexpected_catalog_error(
                    http_response,
                    context.client.disable_header_redaction(),
                )
                .await);
            }
        };

        let metadata_location = response.metadata_location.as_ref().ok_or(Error::new(
            ErrorKind::DataInvalid,
            "Metadata location missing in `create_table` response!",
        ))?;

        let config = response
            .config
            .into_iter()
            .chain(self.user_config.props.clone())
            .collect();

        let file_io = self
            .load_file_io(
                Some(metadata_location),
                Some(config),
                response.storage_credentials.as_deref(),
            )
            .await?;

        let table_builder = Table::builder()
            .identifier(table_ident.clone())
            .file_io(file_io)
            .metadata(response.metadata);

        if let Some(metadata_location) = response.metadata_location {
            table_builder.metadata_location(metadata_location).build()
        } else {
            table_builder.build()
        }
    }

    /// Load table from the catalog.
    ///
    /// The local `RestCatalog` config wins over the server response on a property collision.
    async fn load_table(&self, table_ident: &TableIdent) -> Result<Table> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::GET, context.config.table_endpoint(table_ident))
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        let response = match http_response.status() {
            StatusCode::OK | StatusCode::NOT_MODIFIED => {
                deserialize_catalog_response::<LoadTableResult>(http_response).await?
            }
            StatusCode::NOT_FOUND => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    "Tried to load a table that does not exist",
                ));
            }
            _ => {
                return Err(deserialize_unexpected_catalog_error(
                    http_response,
                    context.client.disable_header_redaction(),
                )
                .await);
            }
        };

        let config = response
            .config
            .into_iter()
            .chain(self.user_config.props.clone())
            .collect();

        let file_io = self
            .load_file_io(
                response.metadata_location.as_deref(),
                Some(config),
                response.storage_credentials.as_deref(),
            )
            .await?;

        let table_builder = Table::builder()
            .identifier(table_ident.clone())
            .file_io(file_io)
            .metadata(response.metadata);

        if let Some(metadata_location) = response.metadata_location {
            table_builder.metadata_location(metadata_location).build()
        } else {
            table_builder.build()
        }
    }

    /// Drop a table from the catalog.
    async fn drop_table(&self, table: &TableIdent) -> Result<()> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::DELETE, context.config.table_endpoint(table))
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        match http_response.status() {
            StatusCode::NO_CONTENT | StatusCode::OK => Ok(()),
            StatusCode::NOT_FOUND => Err(Error::new(
                ErrorKind::Unexpected,
                "Tried to drop a table that does not exist",
            )),
            _ => Err(deserialize_unexpected_catalog_error(
                http_response,
                context.client.disable_header_redaction(),
            )
            .await),
        }
    }

    /// Check if a table exists in the catalog.
    async fn table_exists(&self, table: &TableIdent) -> Result<bool> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::HEAD, context.config.table_endpoint(table))
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        match http_response.status() {
            StatusCode::NO_CONTENT | StatusCode::OK => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            _ => Err(deserialize_unexpected_catalog_error(
                http_response,
                context.client.disable_header_redaction(),
            )
            .await),
        }
    }

    /// Rename a table in the catalog.
    async fn rename_table(&self, src: &TableIdent, dest: &TableIdent) -> Result<()> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::POST, context.config.rename_table_endpoint())
            .json(&RenameTableRequest {
                source: src.clone(),
                destination: dest.clone(),
            })
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        match http_response.status() {
            StatusCode::NO_CONTENT | StatusCode::OK => Ok(()),
            StatusCode::NOT_FOUND => Err(Error::new(
                ErrorKind::Unexpected,
                "Tried to rename a table that does not exist (is the namespace correct?)",
            )),
            StatusCode::CONFLICT => Err(Error::new(
                ErrorKind::Unexpected,
                "Tried to rename a table to a name that already exists",
            )),
            _ => Err(deserialize_unexpected_catalog_error(
                http_response,
                context.client.disable_header_redaction(),
            )
            .await),
        }
    }

    async fn register_table(
        &self,
        table_ident: &TableIdent,
        metadata_location: String,
    ) -> Result<Table> {
        let context = self.context().await?;

        let request = context
            .client
            .request(
                Method::POST,
                context
                    .config
                    .register_table_endpoint(table_ident.namespace()),
            )
            .json(&RegisterTableRequest {
                name: table_ident.name.clone(),
                metadata_location: metadata_location.clone(),
                overwrite: Some(false),
            })
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        let response: LoadTableResult = match http_response.status() {
            StatusCode::OK => {
                deserialize_catalog_response::<LoadTableResult>(http_response).await?
            }
            StatusCode::NOT_FOUND => {
                return Err(Error::new(
                    ErrorKind::NamespaceNotFound,
                    "The namespace specified does not exist.",
                ));
            }
            StatusCode::CONFLICT => {
                return Err(Error::new(
                    ErrorKind::TableAlreadyExists,
                    "The given table already exists.",
                ));
            }
            _ => {
                return Err(deserialize_unexpected_catalog_error(
                    http_response,
                    context.client.disable_header_redaction(),
                )
                .await);
            }
        };

        let metadata_location = response.metadata_location.as_ref().ok_or(Error::new(
            ErrorKind::DataInvalid,
            "Metadata location missing in `register_table` response!",
        ))?;

        let file_io = self
            .load_file_io(
                Some(metadata_location),
                None,
                response.storage_credentials.as_deref(),
            )
            .await?;

        Table::builder()
            .identifier(table_ident.clone())
            .file_io(file_io)
            .metadata(response.metadata)
            .metadata_location(metadata_location.clone())
            .build()
    }

    async fn update_table(&self, mut commit: TableCommit) -> Result<Table> {
        let context = self.context().await?;

        let request = context
            .client
            .request(
                Method::POST,
                context.config.table_endpoint(commit.identifier()),
            )
            .json(&CommitTableRequest {
                identifier: Some(commit.identifier().clone()),
                requirements: commit.take_requirements(),
                updates: commit.take_updates(),
            })
            .build()?;

        // Commit requests classify transport failures sent-vs-unsent: a post-send failure maps
        // to `ErrorKind::CommitStateUnknown` (GAP_MATRIX row R157). See
        // `HttpClient::query_catalog_for_commit`.
        let http_response = context.client.query_catalog_for_commit(request).await?;

        let response: CommitTableResponse = match http_response.status() {
            // A 200 means the commit landed. If the body is then unreadable, surface the
            // unknown-outcome class, so the caller does not re-run a durable commit.
            StatusCode::OK => {
                deserialize_catalog_response(http_response)
                    .await
                    .map_err(|error| {
                        Error::new(
                            ErrorKind::CommitStateUnknown,
                            "The commit request returned HTTP 200 but its response could not be \
                         read; the commit almost certainly landed — verify before retrying.",
                        )
                        .with_source(error)
                    })?
            }
            StatusCode::NOT_FOUND => {
                return Err(Error::new(
                    ErrorKind::TableNotFound,
                    "Tried to update a table that does not exist",
                ));
            }
            StatusCode::CONFLICT => {
                return Err(Error::new(
                    ErrorKind::CatalogCommitConflicts,
                    "CatalogCommitConflicts, one or more requirements failed. The client may retry.",
                )
                .with_retryable(true));
            }
            // Java `ErrorHandlers$CommitErrorHandler` (1.10.0, ErrorHandlers.java L88-104) maps
            // 500/502/503/504 → `CommitStateUnknownException`: the service may have applied the
            // update before failing. Never retryable — retrying a landed commit duplicates it.
            StatusCode::INTERNAL_SERVER_ERROR => {
                return Err(Error::new(
                    ErrorKind::CommitStateUnknown,
                    "An unknown server-side problem occurred; the commit state is unknown.",
                ));
            }
            StatusCode::BAD_GATEWAY => {
                return Err(Error::new(
                    ErrorKind::CommitStateUnknown,
                    "A gateway or proxy received an invalid response from the upstream server; the commit state is unknown.",
                ));
            }
            StatusCode::SERVICE_UNAVAILABLE => {
                return Err(Error::new(
                    ErrorKind::CommitStateUnknown,
                    "The server is currently unavailable; the commit state is unknown.",
                ));
            }
            StatusCode::GATEWAY_TIMEOUT => {
                return Err(Error::new(
                    ErrorKind::CommitStateUnknown,
                    "A server-side gateway timeout occurred; the commit state is unknown.",
                ));
            }
            _ => {
                return Err(deserialize_unexpected_catalog_error(
                    http_response,
                    context.client.disable_header_redaction(),
                )
                .await);
            }
        };

        // `CommitTableResponse` carries no `storage-credentials` (Java's post-commit refreshed
        // FileIO credentials are a separate concern), so no vended overlay applies here.
        let file_io = self
            .load_file_io(Some(&response.metadata_location), None, None)
            .await?;

        Table::builder()
            .identifier(commit.identifier().clone())
            .file_io(file_io)
            .metadata(response.metadata)
            .metadata_location(response.metadata_location)
            .build()
    }

    // ========================================================================
    // View surface — mirrors the Iceberg REST view routes / Java `RESTSessionCatalog`'s
    // `RESTViewBuilder` + `RESTViewOperations`. Endpoints: `GET/POST /namespaces/{ns}/views`,
    // `GET/POST/DELETE/HEAD /namespaces/{ns}/views/{view}`, `POST /views/rename`.
    // ========================================================================

    async fn list_views(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        let context = self.context().await?;
        let endpoint = context.config.views_endpoint(namespace);
        let mut identifiers = Vec::new();
        let mut next_token = None;

        loop {
            let mut request = context.client.request(Method::GET, endpoint.clone());
            if let Some(token) = next_token {
                request = request.query(&[("pageToken", token)]);
            }

            let http_response = context.client.query_catalog(request.build()?).await?;

            match http_response.status() {
                StatusCode::OK => {
                    // Java reuses the `ListTablesResponse` shape (`identifiers` / `next-page-token`)
                    // for the list-views response.
                    let response =
                        deserialize_catalog_response::<ListTablesResponse>(http_response).await?;
                    identifiers.extend(response.identifiers);
                    match response.next_page_token {
                        Some(token) => next_token = Some(token),
                        None => break,
                    }
                }
                StatusCode::NOT_FOUND => {
                    return Err(Error::new(
                        ErrorKind::NamespaceNotFound,
                        "Tried to list views of a namespace that does not exist",
                    ));
                }
                _ => {
                    return Err(deserialize_unexpected_catalog_error(
                        http_response,
                        context.client.disable_header_redaction(),
                    )
                    .await);
                }
            }
        }

        Ok(identifiers)
    }

    async fn create_view(
        &self,
        namespace: &NamespaceIdent,
        creation: ViewCreation,
    ) -> Result<View> {
        let context = self.context().await?;

        let view_ident = TableIdent::new(namespace.clone(), creation.name.clone());

        // Build the initial `ViewVersion` from the creation (Java `RESTViewBuilder.create` mints
        // version 1). The server re-assigns the version id and timestamp authoritatively.
        let view_version = iceberg::spec::ViewVersion::builder()
            .with_version_id(1)
            .with_timestamp_ms(chrono::Utc::now().timestamp_millis())
            .with_schema_id(creation.schema.schema_id())
            .with_default_namespace(creation.default_namespace.clone())
            .with_default_catalog(creation.default_catalog.clone())
            .with_summary(creation.summary.clone())
            .with_representations(creation.representations.clone())
            .build();

        let request = context
            .client
            .request(Method::POST, context.config.views_endpoint(namespace))
            .json(&CreateViewRequest {
                name: creation.name,
                location: Some(creation.location),
                view_version,
                schema: creation.schema,
                properties: creation.properties,
            })
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        let response = match http_response.status() {
            StatusCode::OK => deserialize_catalog_response::<LoadViewResult>(http_response).await?,
            StatusCode::NOT_FOUND => {
                return Err(Error::new(
                    ErrorKind::NamespaceNotFound,
                    "Tried to create a view under a namespace that does not exist",
                ));
            }
            StatusCode::CONFLICT => {
                return Err(Error::new(
                    ErrorKind::ViewAlreadyExists,
                    "The view already exists",
                ));
            }
            _ => {
                return Err(deserialize_unexpected_catalog_error(
                    http_response,
                    context.client.disable_header_redaction(),
                )
                .await);
            }
        };

        self.build_view_from_load_result(view_ident, response).await
    }

    async fn load_view(&self, view_ident: &TableIdent) -> Result<View> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::GET, context.config.view_endpoint(view_ident))
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        let response = match http_response.status() {
            StatusCode::OK => deserialize_catalog_response::<LoadViewResult>(http_response).await?,
            StatusCode::NOT_FOUND => {
                return Err(Error::new(
                    ErrorKind::ViewNotFound,
                    "Tried to load a view that does not exist",
                ));
            }
            _ => {
                return Err(deserialize_unexpected_catalog_error(
                    http_response,
                    context.client.disable_header_redaction(),
                )
                .await);
            }
        };

        self.build_view_from_load_result(view_ident.clone(), response)
            .await
    }

    async fn drop_view(&self, view: &TableIdent) -> Result<()> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::DELETE, context.config.view_endpoint(view))
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        match http_response.status() {
            StatusCode::NO_CONTENT | StatusCode::OK => Ok(()),
            StatusCode::NOT_FOUND => Err(Error::new(
                ErrorKind::ViewNotFound,
                "Tried to drop a view that does not exist",
            )),
            _ => Err(deserialize_unexpected_catalog_error(
                http_response,
                context.client.disable_header_redaction(),
            )
            .await),
        }
    }

    async fn view_exists(&self, view: &TableIdent) -> Result<bool> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::HEAD, context.config.view_endpoint(view))
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        match http_response.status() {
            StatusCode::NO_CONTENT | StatusCode::OK => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            _ => Err(deserialize_unexpected_catalog_error(
                http_response,
                context.client.disable_header_redaction(),
            )
            .await),
        }
    }

    async fn rename_view(&self, src: &TableIdent, dest: &TableIdent) -> Result<()> {
        let context = self.context().await?;

        let request = context
            .client
            .request(Method::POST, context.config.rename_view_endpoint())
            .json(&RenameTableRequest {
                source: src.clone(),
                destination: dest.clone(),
            })
            .build()?;

        let http_response = context.client.query_catalog(request).await?;

        match http_response.status() {
            StatusCode::NO_CONTENT | StatusCode::OK => Ok(()),
            StatusCode::NOT_FOUND => Err(Error::new(
                ErrorKind::ViewNotFound,
                "Tried to rename a view that does not exist (is the namespace correct?)",
            )),
            StatusCode::CONFLICT => Err(Error::new(
                ErrorKind::ViewAlreadyExists,
                "Tried to rename a view to a name that already exists",
            )),
            _ => Err(deserialize_unexpected_catalog_error(
                http_response,
                context.client.disable_header_redaction(),
            )
            .await),
        }
    }

    async fn update_view(&self, mut commit: ViewCommit) -> Result<View> {
        let context = self.context().await?;

        let view_ident = commit.identifier().clone();

        let request = context
            .client
            .request(Method::POST, context.config.view_endpoint(&view_ident))
            .json(&CommitViewRequest {
                identifier: Some(view_ident.clone()),
                requirements: commit.take_requirements(),
                updates: commit.take_updates(),
            })
            .build()?;

        // Commit requests classify transport failures sent-vs-unsent — mirror the table-side
        // `update_table` posture exactly (GAP_MATRIX row R157).
        let http_response = context.client.query_catalog_for_commit(request).await?;

        let response: LoadViewResult = match http_response.status() {
            // 200 = the view commit LANDED; a lost/unparsable response body must still not
            // send the caller into a blind re-run — surface the unknown-outcome class.
            StatusCode::OK => {
                deserialize_catalog_response(http_response)
                    .await
                    .map_err(|error| {
                        Error::new(
                            ErrorKind::CommitStateUnknown,
                            "The view commit request returned HTTP 200 but its response could not \
                         be read; the commit almost certainly landed — verify before retrying.",
                        )
                        .with_source(error)
                    })?
            }
            StatusCode::NOT_FOUND => {
                return Err(Error::new(
                    ErrorKind::ViewNotFound,
                    "Tried to update a view that does not exist",
                ));
            }
            StatusCode::CONFLICT => {
                return Err(Error::new(
                    ErrorKind::CatalogCommitConflicts,
                    "CatalogCommitConflicts, one or more view requirements failed. The client may retry.",
                )
                .with_retryable(true));
            }
            // Java `ErrorHandlers$ViewCommitErrorHandler` maps 5xx to
            // `CommitStateUnknownException`, not to a generic transport error. The commit may
            // have landed. Mirror the table-side `update_table` posture.
            StatusCode::INTERNAL_SERVER_ERROR => {
                return Err(Error::new(
                    ErrorKind::CommitStateUnknown,
                    "An unknown server-side problem occurred; the commit state is unknown.",
                ));
            }
            StatusCode::BAD_GATEWAY => {
                return Err(Error::new(
                    ErrorKind::CommitStateUnknown,
                    "A gateway or proxy received an invalid response from the upstream server; the commit state is unknown.",
                ));
            }
            StatusCode::SERVICE_UNAVAILABLE => {
                return Err(Error::new(
                    ErrorKind::CommitStateUnknown,
                    "The server is currently unavailable; the commit state is unknown.",
                ));
            }
            StatusCode::GATEWAY_TIMEOUT => {
                return Err(Error::new(
                    ErrorKind::CommitStateUnknown,
                    "A server-side gateway timeout occurred; the commit state is unknown.",
                ));
            }
            _ => {
                return Err(deserialize_unexpected_catalog_error(
                    http_response,
                    context.client.disable_header_redaction(),
                )
                .await);
            }
        };

        self.build_view_from_load_result(view_ident, response).await
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;
    use std::sync::Arc;

    use chrono::{TimeZone, Utc};
    use iceberg::io::LocalFsStorageFactory;
    use iceberg::spec::{
        FormatVersion, NestedField, NullOrder, Operation, PrimitiveType, Schema, Snapshot,
        SnapshotLog, SortDirection, SortField, SortOrder, Summary, Transform, Type,
        UnboundPartitionField, UnboundPartitionSpec,
    };
    use iceberg::transaction::{ApplyTransactionAction, Transaction};
    use mockito::{Mock, Server, ServerGuard};
    use serde_json::json;
    use uuid::uuid;

    use super::*;

    #[tokio::test]
    async fn test_update_config() {
        let mut server = Server::new_async().await;

        let config_mock = server
            .mock("GET", "/v1/config")
            .with_status(200)
            .with_body(
                r#"{
                "overrides": {
                    "warehouse": "s3://iceberg-catalog"
                },
                "defaults": {}
            }"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        assert_eq!(
            catalog
                .context()
                .await
                .unwrap()
                .config
                .props
                .get("warehouse"),
            Some(&"s3://iceberg-catalog".to_string())
        );

        config_mock.assert_async().await;
    }

    async fn create_config_mock(server: &mut ServerGuard) -> Mock {
        server
            .mock("GET", "/v1/config")
            .with_status(200)
            .with_body(
                r#"{
                "overrides": {
                    "warehouse": "s3://iceberg-catalog"
                },
                "defaults": {}
            }"#,
            )
            .create_async()
            .await
    }

    async fn create_oauth_mock(server: &mut ServerGuard) -> Mock {
        create_oauth_mock_with_path(server, "/v1/oauth/tokens", "ey000000000000", 200).await
    }

    async fn create_oauth_mock_with_path(
        server: &mut ServerGuard,
        path: &str,
        token: &str,
        status: usize,
    ) -> Mock {
        let body = format!(
            r#"{{
                "access_token": "{token}",
                "token_type": "Bearer",
                "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "expires_in": 86400
            }}"#
        );
        server
            .mock("POST", path)
            .with_status(status)
            .with_body(body)
            .expect(1)
            .create_async()
            .await
    }

    #[tokio::test]
    async fn test_oauth() {
        let mut server = Server::new_async().await;
        let oauth_mock = create_oauth_mock(&mut server).await;
        let config_mock = create_config_mock(&mut server).await;

        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder()
                .uri(server.url())
                .props(props)
                .build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let token = catalog.context().await.unwrap().client.token().await;
        oauth_mock.assert_async().await;
        config_mock.assert_async().await;
        assert_eq!(token, Some("ey000000000000".to_string()));
    }

    /// A 200-OK token body carries the `access_token`, so a parse error must not quote it.
    /// The test feeds a 200 body that holds a sentinel but is invalid as a `TokenResponse`.
    /// Discriminates the mutation that reattaches the body with `.with_context("json", …)`.
    #[tokio::test]
    async fn test_oauth_token_body_not_leaked_in_error() {
        const TOKEN_SENTINEL: &str = "SUPER_SECRET_ACCESS_TOKEN_DO_NOT_LEAK";

        let mut server = Server::new_async().await;
        // 200 OK, but the JSON shape is wrong for TokenResponse (access_token is a number),
        // so deserialization fails while the secret sentinel is present in the body.
        let oauth_mock = server
            .mock("POST", "/v1/oauth/tokens")
            .with_status(200)
            .with_body(format!(
                r#"{{"access_token": 12345, "note": "{TOKEN_SENTINEL}"}}"#
            ))
            .expect(1)
            .create_async()
            .await;

        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());

        let client = HttpClient::new(
            &RestCatalogConfig::builder()
                .uri(server.url())
                .props(props)
                .build(),
        )
        .unwrap();

        let err = client
            .exchange_credential_for_token()
            .await
            .expect_err("malformed token response must yield an error");

        oauth_mock.assert_async().await;

        let rendered = format!("{err}");
        let debug = format!("{err:?}");
        assert!(
            !rendered.contains(TOKEN_SENTINEL),
            "token-response body leaked into Display: {rendered}"
        );
        assert!(
            !debug.contains(TOKEN_SENTINEL),
            "token-response body leaked into Debug context: {debug}"
        );
    }

    /// Same guard on the non-2xx branch of `exchange_credential_for_token`. A 400 body can
    /// echo the submitted credential, so an `ErrorResponse` parse failure must not quote it.
    /// Discriminates the mutation that reattaches the body with `.with_context("json", …)`.
    #[tokio::test]
    async fn test_oauth_token_error_body_not_leaked_in_error() {
        const CRED_SENTINEL: &str = "SUBMITTED_CLIENT_SECRET_DO_NOT_LEAK";

        let mut server = Server::new_async().await;
        // 400, valid JSON but not a valid `ErrorResponse` (no `error` object), so parsing
        // fails while the secret sentinel is present in the body.
        let oauth_mock = server
            .mock("POST", "/v1/oauth/tokens")
            .with_status(400)
            .with_body(format!(r#"{{"echoed_credential": "{CRED_SENTINEL}"}}"#))
            .expect(1)
            .create_async()
            .await;

        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());

        let client = HttpClient::new(
            &RestCatalogConfig::builder()
                .uri(server.url())
                .props(props)
                .build(),
        )
        .unwrap();

        let err = client
            .exchange_credential_for_token()
            .await
            .expect_err("malformed token error response must yield an error");

        oauth_mock.assert_async().await;

        let rendered = format!("{err}");
        let debug = format!("{err:?}");
        assert!(
            !rendered.contains(CRED_SENTINEL),
            "token error body leaked into Display: {rendered}"
        );
        assert!(
            !debug.contains(CRED_SENTINEL),
            "token error body leaked into Debug context: {debug}"
        );
    }

    #[tokio::test]
    async fn test_oauth_with_optional_param() {
        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());
        props.insert("scope".to_string(), "custom_scope".to_string());
        props.insert("audience".to_string(), "custom_audience".to_string());
        props.insert("resource".to_string(), "custom_resource".to_string());

        let mut server = Server::new_async().await;
        let oauth_mock = server
            .mock("POST", "/v1/oauth/tokens")
            .match_body(mockito::Matcher::Regex("scope=custom_scope".to_string()))
            .match_body(mockito::Matcher::Regex(
                "audience=custom_audience".to_string(),
            ))
            .match_body(mockito::Matcher::Regex(
                "resource=custom_resource".to_string(),
            ))
            .with_status(200)
            .with_body(
                r#"{
                "access_token": "ey000000000000",
                "token_type": "Bearer",
                "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "expires_in": 86400
                }"#,
            )
            .expect(1)
            .create_async()
            .await;

        let config_mock = create_config_mock(&mut server).await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder()
                .uri(server.url())
                .props(props)
                .build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let token = catalog.context().await.unwrap().client.token().await;

        oauth_mock.assert_async().await;
        config_mock.assert_async().await;
        assert_eq!(token, Some("ey000000000000".to_string()));
    }

    #[tokio::test]
    async fn test_invalidate_token() {
        let mut server = Server::new_async().await;
        let oauth_mock = create_oauth_mock(&mut server).await;
        let config_mock = create_config_mock(&mut server).await;

        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder()
                .uri(server.url())
                .props(props)
                .build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let token = catalog.context().await.unwrap().client.token().await;
        oauth_mock.assert_async().await;
        config_mock.assert_async().await;
        assert_eq!(token, Some("ey000000000000".to_string()));

        let oauth_mock =
            create_oauth_mock_with_path(&mut server, "/v1/oauth/tokens", "ey000000000001", 200)
                .await;
        catalog.invalidate_token().await.unwrap();
        let token = catalog.context().await.unwrap().client.token().await;
        oauth_mock.assert_async().await;
        assert_eq!(token, Some("ey000000000001".to_string()));
    }

    #[tokio::test]
    async fn test_invalidate_token_failing_request() {
        let mut server = Server::new_async().await;
        let oauth_mock = create_oauth_mock(&mut server).await;
        let config_mock = create_config_mock(&mut server).await;

        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder()
                .uri(server.url())
                .props(props)
                .build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let token = catalog.context().await.unwrap().client.token().await;
        oauth_mock.assert_async().await;
        config_mock.assert_async().await;
        assert_eq!(token, Some("ey000000000000".to_string()));

        let oauth_mock =
            create_oauth_mock_with_path(&mut server, "/v1/oauth/tokens", "ey000000000001", 500)
                .await;
        catalog.invalidate_token().await.unwrap();
        let token = catalog.context().await.unwrap().client.token().await;
        oauth_mock.assert_async().await;
        assert_eq!(token, None);
    }

    #[tokio::test]
    async fn test_regenerate_token() {
        let mut server = Server::new_async().await;
        let oauth_mock = create_oauth_mock(&mut server).await;
        let config_mock = create_config_mock(&mut server).await;

        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder()
                .uri(server.url())
                .props(props)
                .build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let token = catalog.context().await.unwrap().client.token().await;
        oauth_mock.assert_async().await;
        config_mock.assert_async().await;
        assert_eq!(token, Some("ey000000000000".to_string()));

        let oauth_mock =
            create_oauth_mock_with_path(&mut server, "/v1/oauth/tokens", "ey000000000001", 200)
                .await;
        catalog.regenerate_token().await.unwrap();
        oauth_mock.assert_async().await;
        let token = catalog.context().await.unwrap().client.token().await;
        assert_eq!(token, Some("ey000000000001".to_string()));
    }

    #[tokio::test]
    async fn test_regenerate_token_failing_request() {
        let mut server = Server::new_async().await;
        let oauth_mock = create_oauth_mock(&mut server).await;
        let config_mock = create_config_mock(&mut server).await;

        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder()
                .uri(server.url())
                .props(props)
                .build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let token = catalog.context().await.unwrap().client.token().await;
        oauth_mock.assert_async().await;
        config_mock.assert_async().await;
        assert_eq!(token, Some("ey000000000000".to_string()));

        let oauth_mock =
            create_oauth_mock_with_path(&mut server, "/v1/oauth/tokens", "ey000000000001", 500)
                .await;
        let invalidate_result = catalog.regenerate_token().await;
        assert!(invalidate_result.is_err());
        oauth_mock.assert_async().await;
        let token = catalog.context().await.unwrap().client.token().await;

        // original token is left intact
        assert_eq!(token, Some("ey000000000000".to_string()));
    }

    #[tokio::test]
    async fn test_http_headers() {
        let server = Server::new_async().await;
        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());

        let config = RestCatalogConfig::builder()
            .uri(server.url())
            .props(props)
            .build();
        let headers: HeaderMap = config.extra_headers().unwrap();

        let expected_headers = HeaderMap::from_iter([
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            ),
            (
                HeaderName::from_static("x-client-version"),
                HeaderValue::from_static(ICEBERG_REST_SPEC_VERSION),
            ),
            (
                header::USER_AGENT,
                HeaderValue::from_str(&format!("iceberg-rs/{CARGO_PKG_VERSION}")).unwrap(),
            ),
        ]);
        assert_eq!(headers, expected_headers);
    }

    #[tokio::test]
    async fn test_http_headers_with_custom_headers() {
        let server = Server::new_async().await;
        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());
        props.insert(
            "header.content-type".to_string(),
            "application/yaml".to_string(),
        );
        props.insert(
            "header.customized-header".to_string(),
            "some/value".to_string(),
        );

        let config = RestCatalogConfig::builder()
            .uri(server.url())
            .props(props)
            .build();
        let headers: HeaderMap = config.extra_headers().unwrap();

        let expected_headers = HeaderMap::from_iter([
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/yaml"),
            ),
            (
                HeaderName::from_static("x-client-version"),
                HeaderValue::from_static(ICEBERG_REST_SPEC_VERSION),
            ),
            (
                header::USER_AGENT,
                HeaderValue::from_str(&format!("iceberg-rs/{CARGO_PKG_VERSION}")).unwrap(),
            ),
            (
                HeaderName::from_static("customized-header"),
                HeaderValue::from_static("some/value"),
            ),
        ]);
        assert_eq!(headers, expected_headers);
    }

    #[tokio::test]
    async fn test_oauth_with_oauth2_server_uri() {
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;

        let mut auth_server = Server::new_async().await;
        let auth_server_path = "/some/path";
        let oauth_mock =
            create_oauth_mock_with_path(&mut auth_server, auth_server_path, "ey000000000000", 200)
                .await;

        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());
        props.insert(
            "oauth2-server-uri".to_string(),
            format!("{}{}", auth_server.url(), auth_server_path).to_string(),
        );

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder()
                .uri(server.url())
                .props(props)
                .build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let token = catalog.context().await.unwrap().client.token().await;

        oauth_mock.assert_async().await;
        config_mock.assert_async().await;
        assert_eq!(token, Some("ey000000000000".to_string()));
    }

    #[tokio::test]
    async fn test_config_override() {
        let mut server = Server::new_async().await;
        let mut redirect_server = Server::new_async().await;
        let new_uri = redirect_server.url();

        let config_mock = server
            .mock("GET", "/v1/config")
            .with_status(200)
            .with_body(
                json!(
                    {
                        "overrides": {
                            "uri": new_uri,
                            "warehouse": "s3://iceberg-catalog",
                            "prefix": "ice/warehouses/my"
                        },
                        "defaults": {},
                    }
                )
                .to_string(),
            )
            .create_async()
            .await;

        let list_ns_mock = redirect_server
            .mock("GET", "/v1/ice/warehouses/my/namespaces")
            .with_body(
                r#"{
                    "namespaces": []
                }"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let _namespaces = catalog.list_namespaces(None).await.unwrap();

        config_mock.assert_async().await;
        list_ns_mock.assert_async().await;
    }

    /// `disable-header-redaction` is client-only, so a server must not flip redaction off
    /// through `/v1/config`. Discriminates dropping the guard in `merge_with_config`.
    #[test]
    fn test_server_cannot_disable_header_redaction() {
        // User did NOT opt into disabling redaction (so redaction is ON by default).
        let user = RestCatalogConfig::builder()
            .uri("http://localhost".to_string())
            .build();
        assert!(!user.disable_header_redaction());

        // Hostile server tries to disable redaction via both override channels.
        let mut overrides = HashMap::new();
        overrides.insert(
            REST_CATALOG_PROP_DISABLE_HEADER_REDACTION.to_string(),
            "true".to_string(),
        );
        let mut defaults = HashMap::new();
        defaults.insert(
            REST_CATALOG_PROP_DISABLE_HEADER_REDACTION.to_string(),
            "true".to_string(),
        );
        let server_config = CatalogConfig {
            overrides,
            defaults,
        };

        let merged = user.merge_with_config(server_config);

        // Redaction must remain ON — the server's flip is rejected.
        assert!(
            !merged.disable_header_redaction(),
            "server overrides/defaults must not be able to disable header redaction"
        );
    }

    /// SECURITY (Fix 2): the legitimate case — when the *user's own* config opts in, the
    /// setting is honoured (and the server cannot un-set it either).
    #[test]
    fn test_user_can_disable_header_redaction() {
        let mut props = HashMap::new();
        props.insert(
            REST_CATALOG_PROP_DISABLE_HEADER_REDACTION.to_string(),
            "true".to_string(),
        );
        let user = RestCatalogConfig::builder()
            .uri("http://localhost".to_string())
            .props(props)
            .build();
        assert!(user.disable_header_redaction());

        // Server tries to silently re-enable redaction; the user's choice still wins.
        let mut overrides = HashMap::new();
        overrides.insert(
            REST_CATALOG_PROP_DISABLE_HEADER_REDACTION.to_string(),
            "false".to_string(),
        );
        let server_config = CatalogConfig {
            overrides,
            defaults: HashMap::new(),
        };

        let merged = user.merge_with_config(server_config);

        assert!(
            merged.disable_header_redaction(),
            "the user's own disable-header-redaction choice must be honoured"
        );
    }

    /// `RestCatalogConfig`'s `Debug` must redact `credential` and `token`.
    /// Discriminates the mutation that returns to `#[derive(Debug)]`.
    #[test]
    fn test_rest_catalog_config_debug_redacts_credential() {
        const CREDENTIAL_SENTINEL: &str = "client_id_DO_NOT_LEAK:client_secret_DO_NOT_LEAK";
        const TOKEN_SENTINEL: &str = "BEARER_TOKEN_DO_NOT_LEAK";

        let mut props = HashMap::new();
        props.insert("credential".to_string(), CREDENTIAL_SENTINEL.to_string());
        props.insert("token".to_string(), TOKEN_SENTINEL.to_string());
        props.insert("warehouse-style".to_string(), "visible-value".to_string());

        let config = RestCatalogConfig::builder()
            .uri("http://localhost".to_string())
            .props(props)
            .build();

        let debug = format!("{config:?}");

        // Secret prop values must never appear.
        assert!(
            !debug.contains(CREDENTIAL_SENTINEL),
            "Debug leaked the credential: {debug}"
        );
        assert!(
            !debug.contains(TOKEN_SENTINEL),
            "Debug leaked the token: {debug}"
        );
        // Presence is signalled via the redaction marker, and non-secret props stay visible.
        assert!(debug.contains("***"), "expected redaction marker: {debug}");
        assert!(
            debug.contains("visible-value"),
            "Debug dropped non-secret props: {debug}"
        );
    }

    /// [`RestCatalog::load_file_io`] clones `RestCatalogConfig::props` into the `FileIO` props,
    /// so operators put storage credentials there. `Debug` must redact every key below.
    /// Discriminates the mutation that restores the exact-match `SECRET_PROP_KEYS` list.
    #[test]
    fn test_rest_catalog_config_debug_redacts_vended_filei_o_credentials() {
        const SENTINEL: &str = "SENTINEL_MUST_NOT_APPEAR_IN_DEBUG";

        for key in [
            "s3.secret-access-key",
            "s3.session-token",
            "s3.access-key-id",
            "gcs.oauth2.token",
            "adls.connection-string",
        ] {
            let props = HashMap::from([
                (key.to_string(), SENTINEL.to_string()),
                (
                    "s3.endpoint".to_string(),
                    "https://s3.example.test".to_string(),
                ),
            ]);

            let config = RestCatalogConfig::builder()
                .uri("http://localhost".to_string())
                .props(props)
                .build();

            let debug = format!("{config:?}");

            assert!(
                !debug.contains(SENTINEL),
                "Debug leaked the value of `{key}`: {debug}"
            );
            assert!(
                debug.contains("***"),
                "`{key}` was not masked at all: {debug}"
            );
            // Anti-over-redaction: the key itself and non-secret siblings stay readable.
            assert!(
                debug.contains(key),
                "Debug dropped the key `{key}`: {debug}"
            );
            assert!(
                debug.contains("https://s3.example.test"),
                "Debug dropped the non-secret endpoint alongside `{key}`: {debug}"
            );
        }
    }

    /// `iceberg::Error` renders context verbatim, so a raw response body in the context defeats
    /// the `Debug` redaction on the wire types. Version skew is the realistic trigger.
    ///
    /// Here `/v1/config` returns a vended credential in `defaults` and omits `overrides`, so
    /// `CatalogConfig` fails to deserialize with the secret in the body. Discriminates the
    /// mutation that restores `.with_context("json", …)` in `deserialize_catalog_response`.
    #[tokio::test]
    async fn test_config_parse_failure_does_not_leak_body() {
        const SENTINEL: &str = "SENTINEL_MUST_NOT_APPEAR_IN_ERROR";

        let mut server = Server::new_async().await;
        // 200 OK, secret present in `defaults`, but `overrides` is absent => parse failure.
        let config_mock = server
            .mock("GET", "/v1/config")
            .with_status(200)
            .with_body(format!(
                r#"{{"defaults": {{"s3.secret-access-key": "{SENTINEL}"}}}}"#
            ))
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let err = catalog
            .list_namespaces(None)
            .await
            .expect_err("an unparsable /v1/config body must surface an error");

        config_mock.assert_async().await;

        let rendered = format!("{err}");
        let debug = format!("{err:?}");
        assert!(
            !rendered.contains(SENTINEL),
            "config body leaked into Display: {rendered}"
        );
        assert!(
            !debug.contains(SENTINEL),
            "config body leaked into Debug: {debug}"
        );
        // Anti-over-redaction: the safe diagnostics must still be attached, or the error is
        // useless and this test would pass vacuously against a context-free error.
        assert!(
            rendered.contains("response_body_len"),
            "expected the safe body-length diagnostic: {rendered}"
        );
        assert!(
            rendered.contains("status"),
            "expected the safe status diagnostic: {rendered}"
        );
    }

    /// A `load_table` 200 body carries both the `config` overlay and the vended credentials.
    /// The body here fails to parse because `metadata` has the wrong JSON type.
    ///
    /// Scope: a scalar mismatch only, where `serde_json` echoes just the offending scalar. A
    /// mismatch at a container boundary echoes a whole sub-document; see
    /// [`test_known_residue_double_encoded_body_leaks_through_error_source`].
    /// Discriminates the mutation that restores `.with_context("json", …)`.
    #[tokio::test]
    async fn test_load_table_parse_failure_does_not_leak_vended_credentials() {
        const SENTINEL: &str = "SENTINEL_MUST_NOT_APPEAR_IN_ERROR";

        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;

        // A real load_table body, carrying vended credentials, corrupted so it cannot parse.
        let mut body: serde_json::Value = serde_json::from_str(&load_table_body(
            json!({ "s3.secret-access-key": SENTINEL }),
            Some(json!([{
                "prefix": "s3://warehouse",
                "config": { "s3.session-token": SENTINEL }
            }])),
        ))
        .expect("patched load_table body must be valid JSON");
        body["metadata"] = json!(12345);
        let broken = serde_json::to_string(&body).expect("serialize the corrupted body");
        assert!(
            broken.contains(SENTINEL),
            "precondition: the body under test must actually carry the secret"
        );

        let table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/table1")
            .with_status(200)
            .with_body(broken)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let err = catalog
            .load_table(&TableIdent::new(
                NamespaceIdent::new("ns1".to_string()),
                "table1".to_string(),
            ))
            .await
            .expect_err("an unparsable load_table body must surface an error");

        config_mock.assert_async().await;
        table_mock.assert_async().await;

        let rendered = format!("{err}");
        let debug = format!("{err:?}");
        assert!(
            !rendered.contains(SENTINEL),
            "vended credentials leaked into Display: {rendered}"
        );
        assert!(
            !debug.contains(SENTINEL),
            "vended credentials leaked into Debug: {debug}"
        );
        assert!(
            rendered.contains("response_body_len"),
            "expected the safe body-length diagnostic: {rendered}"
        );
    }

    /// `iceberg::Error` renders the `source` verbatim, so withholding the body from the context
    /// alone leaks nothing only if the source is sanitized too. `serde_json` echoes the value at
    /// the failure position, which at a container boundary is a whole sub-document.
    ///
    /// A gateway that emits a nested object as a JSON string turns `config` into a string holding
    /// the vended-credential map. `SanitizedJsonError` keeps the chain but carries only the
    /// category and position. Discriminates restoring `.with_source(e)` on the `serde_json` error.
    #[tokio::test]
    async fn test_double_encoded_body_does_not_leak_through_error_source() {
        const SENTINEL: &str = "SENTINEL_LEAKS_VIA_SERDE_SOURCE_KNOWN_RESIDUE";

        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;

        // `config` arrives DOUBLE-ENCODED: a JSON string whose content is the credential map.
        // Metadata is untouched and valid, so the ONLY parse failure is the container mismatch.
        let body = load_table_body(
            json!(format!(r#"{{"s3.secret-access-key":"{SENTINEL}"}}"#)),
            None,
        );
        assert!(
            body.contains(SENTINEL),
            "precondition: the body under test must carry the secret"
        );

        let table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/table1")
            .with_status(200)
            .with_body(body)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let err = catalog
            .load_table(&TableIdent::new(
                NamespaceIdent::new("ns1".to_string()),
                "table1".to_string(),
            ))
            .await
            .expect_err("a double-encoded config must fail to deserialize");

        config_mock.assert_async().await;
        table_mock.assert_async().await;

        let rendered = format!("{err}");
        let debug = format!("{err:?}");

        // The raw-body CONTEXT attach is gone — that part of the F1 fix holds.
        assert!(
            rendered.contains("response_body_len"),
            "the safe diagnostics must still be attached: {rendered}"
        );

        // ...and the container-boundary echo no longer reaches the log through `source`.
        assert!(
            !rendered.contains(SENTINEL),
            "the double-encoded config leaked through `source` into Display: {rendered}"
        );
        assert!(
            !debug.contains(SENTINEL),
            "the double-encoded config leaked through `source` into Debug: {debug}"
        );

        // The chain is sanitized, not deleted. Without this assertion the test would pass
        // against an error that simply dropped `with_source`.
        let source = std::error::Error::source(&err)
            .expect("the parse failure must still carry a source — the chain may not be deleted");
        let source_text = source.to_string();
        assert!(
            source_text.contains("json data error"),
            "the sanitized source must still classify the failure: {source_text}"
        );
        assert!(
            source_text.contains("line 1 column "),
            "the sanitized source must still carry the failure position: {source_text}"
        );
        assert!(
            rendered.contains("json data error at line 1 column "),
            "the sanitized source must be rendered into Display: {rendered}"
        );
    }

    /// `deserialize_unexpected_catalog_error` is the fallthrough for the write routes, whose
    /// request types carry operator property maps. A server that echoes the offending request
    /// back therefore returns a body with secrets in it.
    ///
    /// Pins that the attached body is key-redacted: the secret value is masked, the server
    /// `message` and every non-secret property survive. Discriminates the mutation that restores
    /// `.with_context("json", String::from_utf8_lossy(&bytes))`.
    #[tokio::test]
    async fn test_non_2xx_body_masks_echoed_secret_properties() {
        const SENTINEL: &str = "SENTINEL_ECHOED_BACK_BY_THE_SERVER";

        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;

        // A 422 whose payload echoes the submitted property map back at the client.
        let body = json!({
            "error": {
                "message": "Cannot create namespace: invalid property value",
                "type": "BadRequestException",
                "code": 422,
                "submitted": {
                    "properties": {
                        "s3.secret-access-key": SENTINEL,
                        "owner": "analytics-team"
                    }
                }
            }
        })
        .to_string();
        assert!(
            body.contains(SENTINEL),
            "precondition: the echoed body must actually carry the secret"
        );

        let ns_mock = server
            .mock("POST", "/v1/namespaces")
            .with_status(422)
            .with_body(body)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let err = catalog
            .create_namespace(
                &NamespaceIdent::new("ns1".to_string()),
                HashMap::from([("s3.secret-access-key".to_string(), SENTINEL.to_string())]),
            )
            .await
            .expect_err("a 422 must surface an error");

        config_mock.assert_async().await;
        ns_mock.assert_async().await;

        let rendered = format!("{err}");
        let debug = format!("{err:?}");
        assert!(
            !rendered.contains(SENTINEL),
            "the echoed secret leaked into Display: {rendered}"
        );
        assert!(
            !debug.contains(SENTINEL),
            "the echoed secret leaked into Debug: {debug}"
        );

        // Anti-over-redaction: the diagnostic payload must survive, or the error says nothing and
        // this test would pass vacuously against a body that was simply dropped.
        assert!(
            rendered.contains("Cannot create namespace: invalid property value"),
            "the server's diagnostic message was dropped: {rendered}"
        );
        assert!(
            rendered.contains("BadRequestException"),
            "the server's error type was dropped: {rendered}"
        );
        assert!(
            rendered.contains("s3.secret-access-key"),
            "the secret KEY must stay visible for diagnostics: {rendered}"
        );
        assert!(
            rendered.contains("analytics-team"),
            "a non-secret echoed property was over-redacted: {rendered}"
        );
    }

    /// A non-JSON body cannot be key-redacted, so only its byte length survives. A proxy error
    /// page can quote the request it forwarded. Discriminates restoring
    /// `.with_context("json", String::from_utf8_lossy(&bytes))`.
    #[tokio::test]
    async fn test_non_2xx_non_json_body_is_withheld() {
        const SENTINEL: &str = "SENTINEL_IN_A_GATEWAY_ERROR_PAGE";

        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;

        let body = format!(
            "<html><body>502 Bad Gateway: upstream rejected s3.secret-access-key={SENTINEL}\
             </body></html>"
        );

        let ns_mock = server
            .mock("POST", "/v1/namespaces")
            .with_status(502)
            .with_body(&body)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let err = catalog
            .create_namespace(&NamespaceIdent::new("ns1".to_string()), HashMap::new())
            .await
            .expect_err("a 502 must surface an error");

        config_mock.assert_async().await;
        ns_mock.assert_async().await;

        let rendered = format!("{err}");
        assert!(
            !rendered.contains(SENTINEL),
            "the non-JSON body leaked into Display: {rendered}"
        );
        // Anti-vacuity: presence and size are still reported, and so is the status.
        assert!(
            rendered.contains("<non-JSON body withheld"),
            "the withheld-body marker is missing: {rendered}"
        );
        assert!(
            rendered.contains(&body.len().to_string()),
            "the body length diagnostic is missing: {rendered}"
        );
        assert!(
            rendered.contains("502"),
            "the status diagnostic is missing: {rendered}"
        );
    }

    /// Java applies no host, IP, or scheme restriction to `uri` or `oauth2-server-uri`. A private
    /// endpoint is the normal deployment, so a private-IP blocklist is rejected. See the trust
    /// note on [`REST_CATALOG_PROP_URI`].
    ///
    /// Pins that a loopback, RFC 1918, or link-local endpoint is accepted verbatim.
    /// Discriminates the mutation that adds any private-address or scheme blocklist.
    #[test]
    fn test_private_and_loopback_uris_are_accepted_java_parity() {
        // Catalog uri: accepted verbatim, no rewriting, no rejection.
        for uri in [
            "http://127.0.0.1:8181",
            "http://localhost:8181",
            "http://10.0.0.7:8181",
            "http://192.168.1.10:8181",
            "http://169.254.169.254",
            "http://catalog.internal:8181",
        ] {
            let config = RestCatalogConfig::builder().uri(uri.to_string()).build();
            assert_eq!(
                config.get_token_endpoint(),
                format!("{uri}/v1/oauth/tokens"),
                "catalog uri `{uri}` must be used verbatim (Java applies no address restriction)"
            );
        }

        // oauth2-server-uri: an explicitly configured private token endpoint wins verbatim,
        // mirroring Java `AuthConfig.fromProperties`.
        let config = RestCatalogConfig::builder()
            .uri("https://catalog.example.test".to_string())
            .props(HashMap::from([(
                "oauth2-server-uri".to_string(),
                "http://10.1.2.3:9000/token".to_string(),
            )]))
            .build();
        assert_eq!(
            config.get_token_endpoint(),
            "http://10.1.2.3:9000/token",
            "a configured private oauth2-server-uri must be honoured verbatim"
        );
    }

    #[tokio::test]
    async fn test_list_namespace() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let list_ns_mock = server
            .mock("GET", "/v1/namespaces")
            .with_body(
                r#"{
                "namespaces": [
                    ["ns1", "ns11"],
                    ["ns2"]
                ]
            }"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let namespaces = catalog.list_namespaces(None).await.unwrap();

        let expected_ns = vec![
            NamespaceIdent::from_vec(vec!["ns1".to_string(), "ns11".to_string()]).unwrap(),
            NamespaceIdent::from_vec(vec!["ns2".to_string()]).unwrap(),
        ];

        assert_eq!(expected_ns, namespaces);

        config_mock.assert_async().await;
        list_ns_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_list_namespace_with_pagination() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let list_ns_mock_page1 = server
            .mock("GET", "/v1/namespaces")
            .with_body(
                r#"{
                "namespaces": [
                    ["ns1", "ns11"],
                    ["ns2"]
                ],
                "next-page-token": "token123"
            }"#,
            )
            .create_async()
            .await;

        let list_ns_mock_page2 = server
            .mock("GET", "/v1/namespaces?pageToken=token123")
            .with_body(
                r#"{
                "namespaces": [
                    ["ns3"],
                    ["ns4", "ns41"]
                ]
            }"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let namespaces = catalog.list_namespaces(None).await.unwrap();

        let expected_ns = vec![
            NamespaceIdent::from_vec(vec!["ns1".to_string(), "ns11".to_string()]).unwrap(),
            NamespaceIdent::from_vec(vec!["ns2".to_string()]).unwrap(),
            NamespaceIdent::from_vec(vec!["ns3".to_string()]).unwrap(),
            NamespaceIdent::from_vec(vec!["ns4".to_string(), "ns41".to_string()]).unwrap(),
        ];

        assert_eq!(expected_ns, namespaces);

        config_mock.assert_async().await;
        list_ns_mock_page1.assert_async().await;
        list_ns_mock_page2.assert_async().await;
    }

    #[tokio::test]
    async fn test_list_namespace_with_multiple_pages() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        // Page 1
        let list_ns_mock_page1 = server
            .mock("GET", "/v1/namespaces")
            .with_body(
                r#"{
                "namespaces": [
                    ["ns1", "ns11"],
                    ["ns2"]
                ],
                "next-page-token": "page2"
            }"#,
            )
            .create_async()
            .await;

        // Page 2
        let list_ns_mock_page2 = server
            .mock("GET", "/v1/namespaces?pageToken=page2")
            .with_body(
                r#"{
                "namespaces": [
                    ["ns3"],
                    ["ns4", "ns41"]
                ],
                "next-page-token": "page3"
            }"#,
            )
            .create_async()
            .await;

        // Page 3
        let list_ns_mock_page3 = server
            .mock("GET", "/v1/namespaces?pageToken=page3")
            .with_body(
                r#"{
                "namespaces": [
                    ["ns5", "ns51", "ns511"]
                ],
                "next-page-token": "page4"
            }"#,
            )
            .create_async()
            .await;

        // Page 4
        let list_ns_mock_page4 = server
            .mock("GET", "/v1/namespaces?pageToken=page4")
            .with_body(
                r#"{
                "namespaces": [
                    ["ns6"],
                    ["ns7"]
                ],
                "next-page-token": "page5"
            }"#,
            )
            .create_async()
            .await;

        // Page 5 (final page)
        let list_ns_mock_page5 = server
            .mock("GET", "/v1/namespaces?pageToken=page5")
            .with_body(
                r#"{
                "namespaces": [
                    ["ns8", "ns81"]
                ]
            }"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let namespaces = catalog.list_namespaces(None).await.unwrap();

        let expected_ns = vec![
            NamespaceIdent::from_vec(vec!["ns1".to_string(), "ns11".to_string()]).unwrap(),
            NamespaceIdent::from_vec(vec!["ns2".to_string()]).unwrap(),
            NamespaceIdent::from_vec(vec!["ns3".to_string()]).unwrap(),
            NamespaceIdent::from_vec(vec!["ns4".to_string(), "ns41".to_string()]).unwrap(),
            NamespaceIdent::from_vec(vec![
                "ns5".to_string(),
                "ns51".to_string(),
                "ns511".to_string(),
            ])
            .unwrap(),
            NamespaceIdent::from_vec(vec!["ns6".to_string()]).unwrap(),
            NamespaceIdent::from_vec(vec!["ns7".to_string()]).unwrap(),
            NamespaceIdent::from_vec(vec!["ns8".to_string(), "ns81".to_string()]).unwrap(),
        ];

        assert_eq!(expected_ns, namespaces);

        // Verify all page requests were made
        config_mock.assert_async().await;
        list_ns_mock_page1.assert_async().await;
        list_ns_mock_page2.assert_async().await;
        list_ns_mock_page3.assert_async().await;
        list_ns_mock_page4.assert_async().await;
        list_ns_mock_page5.assert_async().await;
    }

    #[tokio::test]
    async fn test_create_namespace() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let create_ns_mock = server
            .mock("POST", "/v1/namespaces")
            .with_body(
                r#"{
                "namespace": [ "ns1", "ns11"],
                "properties" : {
                    "key1": "value1"
                }
            }"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let namespaces = catalog
            .create_namespace(
                &NamespaceIdent::from_vec(vec!["ns1".to_string(), "ns11".to_string()]).unwrap(),
                HashMap::from([("key1".to_string(), "value1".to_string())]),
            )
            .await
            .unwrap();

        let expected_ns = Namespace::with_properties(
            NamespaceIdent::from_vec(vec!["ns1".to_string(), "ns11".to_string()]).unwrap(),
            HashMap::from([("key1".to_string(), "value1".to_string())]),
        );

        assert_eq!(expected_ns, namespaces);

        config_mock.assert_async().await;
        create_ns_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_get_namespace() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let get_ns_mock = server
            .mock("GET", "/v1/namespaces/ns1")
            .with_body(
                r#"{
                "namespace": [ "ns1"],
                "properties" : {
                    "key1": "value1"
                }
            }"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let namespaces = catalog
            .get_namespace(&NamespaceIdent::new("ns1".to_string()))
            .await
            .unwrap();

        let expected_ns = Namespace::with_properties(
            NamespaceIdent::new("ns1".to_string()),
            HashMap::from([("key1".to_string(), "value1".to_string())]),
        );

        assert_eq!(expected_ns, namespaces);

        config_mock.assert_async().await;
        get_ns_mock.assert_async().await;
    }

    #[tokio::test]
    async fn check_namespace_exists() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let get_ns_mock = server
            .mock("HEAD", "/v1/namespaces/ns1")
            .with_status(204)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        assert!(
            catalog
                .namespace_exists(&NamespaceIdent::new("ns1".to_string()))
                .await
                .unwrap()
        );

        config_mock.assert_async().await;
        get_ns_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_drop_namespace() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let drop_ns_mock = server
            .mock("DELETE", "/v1/namespaces/ns1")
            .with_status(204)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        catalog
            .drop_namespace(&NamespaceIdent::new("ns1".to_string()))
            .await
            .unwrap();

        config_mock.assert_async().await;
        drop_ns_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_list_tables() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let list_tables_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables")
            .with_status(200)
            .with_body(
                r#"{
                "identifiers": [
                    {
                        "namespace": ["ns1"],
                        "name": "table1"
                    },
                    {
                        "namespace": ["ns1"],
                        "name": "table2"
                    }
                ]
            }"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let tables = catalog
            .list_tables(&NamespaceIdent::new("ns1".to_string()))
            .await
            .unwrap();

        let expected_tables = vec![
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table1".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table2".to_string()),
        ];

        assert_eq!(tables, expected_tables);

        config_mock.assert_async().await;
        list_tables_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_list_tables_with_pagination() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let list_tables_mock_page1 = server
            .mock("GET", "/v1/namespaces/ns1/tables")
            .with_status(200)
            .with_body(
                r#"{
                "identifiers": [
                    {
                        "namespace": ["ns1"],
                        "name": "table1"
                    },
                    {
                        "namespace": ["ns1"],
                        "name": "table2"
                    }
                ],
                "next-page-token": "token456"
            }"#,
            )
            .create_async()
            .await;

        let list_tables_mock_page2 = server
            .mock("GET", "/v1/namespaces/ns1/tables?pageToken=token456")
            .with_status(200)
            .with_body(
                r#"{
                "identifiers": [
                    {
                        "namespace": ["ns1"],
                        "name": "table3"
                    },
                    {
                        "namespace": ["ns1"],
                        "name": "table4"
                    }
                ]
            }"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let tables = catalog
            .list_tables(&NamespaceIdent::new("ns1".to_string()))
            .await
            .unwrap();

        let expected_tables = vec![
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table1".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table2".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table3".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table4".to_string()),
        ];

        assert_eq!(tables, expected_tables);

        config_mock.assert_async().await;
        list_tables_mock_page1.assert_async().await;
        list_tables_mock_page2.assert_async().await;
    }

    #[tokio::test]
    async fn test_list_tables_with_multiple_pages() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        // Page 1
        let list_tables_mock_page1 = server
            .mock("GET", "/v1/namespaces/ns1/tables")
            .with_status(200)
            .with_body(
                r#"{
                "identifiers": [
                    {
                        "namespace": ["ns1"],
                        "name": "table1"
                    },
                    {
                        "namespace": ["ns1"],
                        "name": "table2"
                    }
                ],
                "next-page-token": "page2"
            }"#,
            )
            .create_async()
            .await;

        // Page 2
        let list_tables_mock_page2 = server
            .mock("GET", "/v1/namespaces/ns1/tables?pageToken=page2")
            .with_status(200)
            .with_body(
                r#"{
                "identifiers": [
                    {
                        "namespace": ["ns1"],
                        "name": "table3"
                    },
                    {
                        "namespace": ["ns1"],
                        "name": "table4"
                    }
                ],
                "next-page-token": "page3"
            }"#,
            )
            .create_async()
            .await;

        // Page 3
        let list_tables_mock_page3 = server
            .mock("GET", "/v1/namespaces/ns1/tables?pageToken=page3")
            .with_status(200)
            .with_body(
                r#"{
                "identifiers": [
                    {
                        "namespace": ["ns1"],
                        "name": "table5"
                    }
                ],
                "next-page-token": "page4"
            }"#,
            )
            .create_async()
            .await;

        // Page 4
        let list_tables_mock_page4 = server
            .mock("GET", "/v1/namespaces/ns1/tables?pageToken=page4")
            .with_status(200)
            .with_body(
                r#"{
                "identifiers": [
                    {
                        "namespace": ["ns1"],
                        "name": "table6"
                    },
                    {
                        "namespace": ["ns1"],
                        "name": "table7"
                    }
                ],
                "next-page-token": "page5"
            }"#,
            )
            .create_async()
            .await;

        // Page 5 (final page)
        let list_tables_mock_page5 = server
            .mock("GET", "/v1/namespaces/ns1/tables?pageToken=page5")
            .with_status(200)
            .with_body(
                r#"{
                "identifiers": [
                    {
                        "namespace": ["ns1"],
                        "name": "table8"
                    }
                ]
            }"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let tables = catalog
            .list_tables(&NamespaceIdent::new("ns1".to_string()))
            .await
            .unwrap();

        let expected_tables = vec![
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table1".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table2".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table3".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table4".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table5".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table6".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table7".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table8".to_string()),
        ];

        assert_eq!(tables, expected_tables);

        // Verify all page requests were made
        config_mock.assert_async().await;
        list_tables_mock_page1.assert_async().await;
        list_tables_mock_page2.assert_async().await;
        list_tables_mock_page3.assert_async().await;
        list_tables_mock_page4.assert_async().await;
        list_tables_mock_page5.assert_async().await;
    }

    #[tokio::test]
    async fn test_drop_tables() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let delete_table_mock = server
            .mock("DELETE", "/v1/namespaces/ns1/tables/table1")
            .with_status(204)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        catalog
            .drop_table(&TableIdent::new(
                NamespaceIdent::new("ns1".to_string()),
                "table1".to_string(),
            ))
            .await
            .unwrap();

        config_mock.assert_async().await;
        delete_table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_check_table_exists() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let check_table_exists_mock = server
            .mock("HEAD", "/v1/namespaces/ns1/tables/table1")
            .with_status(204)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        assert!(
            catalog
                .table_exists(&TableIdent::new(
                    NamespaceIdent::new("ns1".to_string()),
                    "table1".to_string(),
                ))
                .await
                .unwrap()
        );

        config_mock.assert_async().await;
        check_table_exists_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_rename_table() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let rename_table_mock = server
            .mock("POST", "/v1/tables/rename")
            .with_status(204)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        catalog
            .rename_table(
                &TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table1".to_string()),
                &TableIdent::new(NamespaceIdent::new("ns1".to_string()), "table2".to_string()),
            )
            .await
            .unwrap();

        config_mock.assert_async().await;
        rename_table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_load_table() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let rename_table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/test1")
            .with_status(200)
            .with_body_from_file(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "load_table_response.json"
            ))
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table = catalog
            .load_table(&TableIdent::new(
                NamespaceIdent::new("ns1".to_string()),
                "test1".to_string(),
            ))
            .await
            .unwrap();

        assert_eq!(
            &TableIdent::from_strs(vec!["ns1", "test1"]).unwrap(),
            table.identifier()
        );
        assert_eq!(
            "s3://warehouse/database/table/metadata/00001-5f2f8166-244c-4eae-ac36-384ecdec81fc.gz.metadata.json",
            table.metadata_location().unwrap()
        );
        assert_eq!(FormatVersion::V1, table.metadata().format_version());
        assert_eq!("s3://warehouse/database/table", table.metadata().location());
        assert_eq!(
            uuid!("b55d9dda-6561-423a-8bfc-787980ce421f"),
            table.metadata().uuid()
        );
        assert_eq!(
            Utc.timestamp_millis_opt(1646787054459).unwrap(),
            table.metadata().last_updated_timestamp().unwrap()
        );
        assert_eq!(
            vec![&Arc::new(
                Schema::builder()
                    .with_fields(vec![
                        NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String))
                            .into(),
                    ])
                    .build()
                    .unwrap()
            )],
            table.metadata().schemas_iter().collect::<Vec<_>>()
        );
        assert_eq!(
            &HashMap::from([
                ("owner".to_string(), "bryan".to_string()),
                (
                    "write.metadata.compression-codec".to_string(),
                    "gzip".to_string()
                )
            ]),
            table.metadata().properties()
        );
        assert_eq!(vec![&Arc::new(Snapshot::builder()
            .with_snapshot_id(3497810964824022504)
            .with_timestamp_ms(1646787054459)
            .with_manifest_list("s3://warehouse/database/table/metadata/snap-3497810964824022504-1-c4f68204-666b-4e50-a9df-b10c34bf6b82.avro")
            .with_sequence_number(0)
            .with_schema_id(0)
            .with_summary(Summary {
                operation: Operation::Append,
                additional_properties: HashMap::from_iter([
                    ("spark.app.id", "local-1646787004168"),
                    ("added-data-files", "1"),
                    ("added-records", "1"),
                    ("added-files-size", "697"),
                    ("changed-partition-count", "1"),
                    ("total-records", "1"),
                    ("total-files-size", "697"),
                    ("total-data-files", "1"),
                    ("total-delete-files", "0"),
                    ("total-position-deletes", "0"),
                    ("total-equality-deletes", "0")
                ].iter().map(|p| (p.0.to_string(), p.1.to_string()))),
            }).build()
        )], table.metadata().snapshots().collect::<Vec<_>>());
        assert_eq!(
            &[SnapshotLog {
                timestamp_ms: 1646787054459,
                snapshot_id: 3497810964824022504,
            }],
            table.metadata().history()
        );
        assert_eq!(
            vec![&Arc::new(SortOrder {
                order_id: 0,
                fields: vec![],
            })],
            table.metadata().sort_orders_iter().collect::<Vec<_>>()
        );

        config_mock.assert_async().await;
        rename_table_mock.assert_async().await;
    }

    // ===================================================================================
    // Vended storage credentials (GAP_MATRIX row R160)
    //
    // Mirrors Java `RESTSessionCatalog.newFileIO(SessionContext, Map, List<Credential>)` +
    // `S3FileIO.clientForStoragePath` / `clientByPrefix`: longest-prefix selection against the
    // table's storage path, credential config layered LAST (wins on collision), no-match is a
    // silent skip. These pins are RED under the mutations named in the G4 charter (invert the
    // selection to shortest-prefix, swap the overlay order, or drop the wiring entirely).
    // ===================================================================================

    /// Build a `load_table` response body from the shared testdata metadata, injecting a specific
    /// `config` map and (optionally) a `storage-credentials` array.
    fn load_table_body(
        config: serde_json::Value,
        storage_credentials: Option<serde_json::Value>,
    ) -> String {
        let path = format!(
            "{}/testdata/{}",
            env!("CARGO_MANIFEST_DIR"),
            "load_table_response.json"
        );
        let file = File::open(path).expect("open load_table_response.json testdata");
        let mut body: serde_json::Value =
            serde_json::from_reader(BufReader::new(file)).expect("parse load_table_response.json");
        body["config"] = config;
        if let Some(creds) = storage_credentials {
            body["storage-credentials"] = creds;
        }
        serde_json::to_string(&body).expect("serialize patched load_table response")
    }

    fn s3_credential(prefix: &str) -> StorageCredential {
        StorageCredential {
            prefix: prefix.to_string(),
            config: HashMap::from([
                ("s3.access-key-id".to_string(), "VENDED_KEY_ID".to_string()),
                (
                    "s3.secret-access-key".to_string(),
                    "VENDED_SECRET".to_string(),
                ),
                ("s3.session-token".to_string(), "VENDED_TOKEN".to_string()),
            ]),
        }
    }

    #[test]
    fn test_select_vended_credential_longest_prefix_wins() {
        // Both credentials prefix the storage path, and the longer one wins. List order runs
        // short then long, so a shortest-prefix mutation flips the pick.
        let short = s3_credential("s3://warehouse/database/table");
        let long = s3_credential("s3://warehouse/database/table/metadata");
        let creds = [short, long];
        let path = "s3://warehouse/database/table/metadata/00001-x.metadata.json";

        let picked = select_vended_credential(Some(path), Some(&creds))
            .expect("a credential prefixes the storage path");

        assert_eq!(
            picked.prefix, "s3://warehouse/database/table/metadata",
            "longest prefix must win, got {}",
            picked.prefix
        );
    }

    #[test]
    fn test_select_vended_credential_first_of_equal_length_kept() {
        // Two distinct equal-length prefixes both cover the path; the strictly-greater replacement
        // rule keeps the first in list order (mirrors Java's `> length` guard).
        let first = StorageCredential {
            prefix: "s3://warehouse/database/tab".to_string(),
            config: HashMap::from([("k".to_string(), "first".to_string())]),
        };
        let second = StorageCredential {
            prefix: "s3://warehouse/database/tab".to_string(),
            config: HashMap::from([("k".to_string(), "second".to_string())]),
        };
        let creds = [first, second];
        let picked = select_vended_credential(
            Some("s3://warehouse/database/table/metadata/x.json"),
            Some(&creds),
        )
        .expect("prefix covers path");
        assert_eq!(picked.config.get("k"), Some(&"first".to_string()));
    }

    #[test]
    fn test_select_vended_credential_no_prefix_match_is_none() {
        // No credential prefixes the storage path → None (Java falls back to the base client
        // silently, raising no error).
        let creds = [s3_credential("s3://other-warehouse/db/table")];
        assert!(
            select_vended_credential(
                Some("s3://warehouse/database/table/metadata/x.json"),
                Some(&creds),
            )
            .is_none(),
            "an unmatched prefix must not be selected"
        );
    }

    #[test]
    fn test_select_vended_credential_none_inputs() {
        let creds = [s3_credential("s3://warehouse/database/table")];
        // No storage path → nothing to match against.
        assert!(select_vended_credential(None, Some(&creds)).is_none());
        // No credentials → nothing to select (the zero-credentials regression path).
        assert!(
            select_vended_credential(Some("s3://warehouse/database/table/m.json"), None).is_none()
        );
    }

    #[tokio::test]
    async fn test_load_table_applies_vended_credentials() {
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;

        // The vended credential must win the colliding `s3.access-key-id` and add its own keys,
        // while the non-colliding `s3.region` survives.
        let body = load_table_body(
            json!({
                "s3.access-key-id": "FROM_TABLE_CONFIG",
                "s3.region": "us-west-2",
            }),
            Some(json!([{
                "prefix": "s3://warehouse/database/table",
                "config": {
                    "s3.access-key-id": "VENDED_KEY_ID",
                    "s3.secret-access-key": "VENDED_SECRET",
                    "s3.session-token": "VENDED_TOKEN",
                },
            }])),
        );

        let table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/test1")
            .with_status(200)
            .with_body(body)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table = catalog
            .load_table(&TableIdent::from_strs(vec!["ns1", "test1"]).unwrap())
            .await
            .expect("load_table with vended credentials");

        let props = table.file_io().config().props();
        // Overlay order + collision winner: the vended credential beats the table config.
        assert_eq!(
            props.get("s3.access-key-id"),
            Some(&"VENDED_KEY_ID".to_string()),
            "vended credential must win the collision over table config"
        );
        // Vended-only keys are layered in.
        assert_eq!(
            props.get("s3.secret-access-key"),
            Some(&"VENDED_SECRET".to_string())
        );
        assert_eq!(
            props.get("s3.session-token"),
            Some(&"VENDED_TOKEN".to_string())
        );
        // Non-colliding table config key survives the overlay.
        assert_eq!(props.get("s3.region"), Some(&"us-west-2".to_string()));

        config_mock.assert_async().await;
        table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_load_table_vended_credentials_redacted_in_debug() {
        // Redaction composition: the vended secrets flow into the `FileIO`'s `StorageConfig`,
        // whose `Debug` redacts every secret-bearing key. A `{:?}` of the `FileIO` must never
        // print a vended value, yet must keep the marker and the non-secret keys.
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;

        let body = load_table_body(
            json!({ "s3.region": "us-west-2" }),
            Some(json!([{
                "prefix": "s3://warehouse/database/table",
                "config": {
                    "s3.access-key-id": "VENDED_KEY_ID",
                    "s3.secret-access-key": "VENDED_SECRET",
                    "s3.session-token": "VENDED_TOKEN",
                },
            }])),
        );

        let table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/test1")
            .with_status(200)
            .with_body(body)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table = catalog
            .load_table(&TableIdent::from_strs(vec!["ns1", "test1"]).unwrap())
            .await
            .expect("load_table with vended credentials");

        let debug = format!("{:?}", table.file_io());
        for secret in ["VENDED_KEY_ID", "VENDED_SECRET", "VENDED_TOKEN"] {
            assert!(
                !debug.contains(secret),
                "Debug of FileIO leaked a vended credential value ({secret}): {debug}"
            );
        }
        assert!(
            debug.contains("***"),
            "expected the redaction marker in Debug: {debug}"
        );
        assert!(
            debug.contains("s3.region"),
            "non-secret key must stay visible in Debug: {debug}"
        );

        config_mock.assert_async().await;
        table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_load_table_no_prefix_match_skips_overlay() {
        // The vended credential's prefix does NOT cover the table's storage path → silent skip
        // (Java raises no error; the table still loads on the un-vended base props).
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;

        let body = load_table_body(
            json!({ "s3.access-key-id": "FROM_TABLE_CONFIG" }),
            Some(json!([{
                "prefix": "s3://some-other-warehouse/db/table",
                "config": { "s3.secret-access-key": "VENDED_SECRET" },
            }])),
        );

        let table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/test1")
            .with_status(200)
            .with_body(body)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table = catalog
            .load_table(&TableIdent::from_strs(vec!["ns1", "test1"]).unwrap())
            .await
            .expect("load_table succeeds even when no credential prefix matches");

        let props = table.file_io().config().props();
        assert_eq!(
            props.get("s3.access-key-id"),
            Some(&"FROM_TABLE_CONFIG".to_string()),
            "unmatched credential must not overlay"
        );
        assert!(
            props.get("s3.secret-access-key").is_none(),
            "unmatched credential's keys must not appear"
        );

        config_mock.assert_async().await;
        table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_load_table_without_vended_credentials_regression() {
        // Zero-credentials path: the common case must behave exactly as before — table config
        // overlays the base props and no vended key is injected.
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;

        let body = load_table_body(json!({ "s3.access-key-id": "FROM_TABLE_CONFIG" }), None);

        let table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/test1")
            .with_status(200)
            .with_body(body)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table = catalog
            .load_table(&TableIdent::from_strs(vec!["ns1", "test1"]).unwrap())
            .await
            .expect("load_table without storage-credentials");

        let props = table.file_io().config().props();
        assert_eq!(
            props.get("s3.access-key-id"),
            Some(&"FROM_TABLE_CONFIG".to_string()),
            "table config must still overlay the base props"
        );
        assert!(
            props.get("s3.secret-access-key").is_none(),
            "no vended keys when the response omits storage-credentials"
        );

        config_mock.assert_async().await;
        table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_load_table_404() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let rename_table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/test1")
            .with_status(404)
            .with_body(r#"
{
    "error": {
        "message": "Table does not exist: ns1.test1 in warehouse 8bcb0838-50fc-472d-9ddb-8feb89ef5f1e",
        "type": "NoSuchNamespaceErrorException",
        "code": 404
    }
}
            "#)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table = catalog
            .load_table(&TableIdent::new(
                NamespaceIdent::new("ns1".to_string()),
                "test1".to_string(),
            ))
            .await;

        assert!(table.is_err());
        assert!(table.err().unwrap().message().contains("does not exist"));

        config_mock.assert_async().await;
        rename_table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_create_table() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let create_table_mock = server
            .mock("POST", "/v1/namespaces/ns1/tables")
            .with_status(200)
            .with_body_from_file(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "create_table_response.json"
            ))
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table_creation = TableCreation::builder()
            .name("test1".to_string())
            .schema(
                Schema::builder()
                    .with_fields(vec![
                        NestedField::optional(1, "foo", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::required(2, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(3, "baz", Type::Primitive(PrimitiveType::Boolean))
                            .into(),
                    ])
                    .with_schema_id(1)
                    .with_identifier_field_ids(vec![2])
                    .build()
                    .unwrap(),
            )
            .properties(HashMap::from([("owner".to_string(), "testx".to_string())]))
            .partition_spec(
                UnboundPartitionSpec::builder()
                    .add_partition_fields(vec![
                        UnboundPartitionField::builder()
                            .source_id(1)
                            .transform(Transform::Truncate(3))
                            .name("id".to_string())
                            .build(),
                    ])
                    .unwrap()
                    .build(),
            )
            .sort_order(
                SortOrder::builder()
                    .with_sort_field(
                        SortField::builder()
                            .source_id(2)
                            .transform(Transform::Identity)
                            .direction(SortDirection::Ascending)
                            .null_order(NullOrder::First)
                            .build(),
                    )
                    .build_unbound()
                    .unwrap(),
            )
            .build();

        let table = catalog
            .create_table(&NamespaceIdent::from_strs(["ns1"]).unwrap(), table_creation)
            .await
            .unwrap();

        assert_eq!(
            &TableIdent::from_strs(vec!["ns1", "test1"]).unwrap(),
            table.identifier()
        );
        assert_eq!(
            "s3://warehouse/database/table/metadata.json",
            table.metadata_location().unwrap()
        );
        assert_eq!(FormatVersion::V1, table.metadata().format_version());
        assert_eq!("s3://warehouse/database/table", table.metadata().location());
        assert_eq!(
            uuid!("bf289591-dcc0-4234-ad4f-5c3eed811a29"),
            table.metadata().uuid()
        );
        assert_eq!(
            1657810967051,
            table
                .metadata()
                .last_updated_timestamp()
                .unwrap()
                .timestamp_millis()
        );
        assert_eq!(
            vec![&Arc::new(
                Schema::builder()
                    .with_fields(vec![
                        NestedField::optional(1, "foo", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::required(2, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(3, "baz", Type::Primitive(PrimitiveType::Boolean))
                            .into(),
                    ])
                    .with_schema_id(0)
                    .with_identifier_field_ids(vec![2])
                    .build()
                    .unwrap()
            )],
            table.metadata().schemas_iter().collect::<Vec<_>>()
        );
        assert_eq!(
            &HashMap::from([
                (
                    "write.delete.parquet.compression-codec".to_string(),
                    "zstd".to_string()
                ),
                (
                    "write.metadata.compression-codec".to_string(),
                    "gzip".to_string()
                ),
                (
                    "write.summary.partition-limit".to_string(),
                    "100".to_string()
                ),
                (
                    "write.parquet.compression-codec".to_string(),
                    "zstd".to_string()
                ),
            ]),
            table.metadata().properties()
        );
        assert!(table.metadata().current_snapshot().is_none());
        assert!(table.metadata().history().is_empty());
        assert_eq!(
            vec![&Arc::new(SortOrder {
                order_id: 0,
                fields: vec![],
            })],
            table.metadata().sort_orders_iter().collect::<Vec<_>>()
        );

        config_mock.assert_async().await;
        create_table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_create_table_409() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let create_table_mock = server
            .mock("POST", "/v1/namespaces/ns1/tables")
            .with_status(409)
            .with_body(r#"
{
    "error": {
        "message": "Table already exists: ns1.test1 in warehouse 8bcb0838-50fc-472d-9ddb-8feb89ef5f1e",
        "type": "AlreadyExistsException",
        "code": 409
    }
}
            "#)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table_creation = TableCreation::builder()
            .name("test1".to_string())
            .schema(
                Schema::builder()
                    .with_fields(vec![
                        NestedField::optional(1, "foo", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::required(2, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(3, "baz", Type::Primitive(PrimitiveType::Boolean))
                            .into(),
                    ])
                    .with_schema_id(1)
                    .with_identifier_field_ids(vec![2])
                    .build()
                    .unwrap(),
            )
            .properties(HashMap::from([("owner".to_string(), "testx".to_string())]))
            .build();

        let table_result = catalog
            .create_table(&NamespaceIdent::from_strs(["ns1"]).unwrap(), table_creation)
            .await;

        assert!(table_result.is_err());
        assert!(
            table_result
                .err()
                .unwrap()
                .message()
                .contains("already exists")
        );

        config_mock.assert_async().await;
        create_table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_update_table() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let load_table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/test1")
            .with_status(200)
            .with_body_from_file(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "load_table_response.json"
            ))
            .create_async()
            .await;

        let update_table_mock = server
            .mock("POST", "/v1/namespaces/ns1/tables/test1")
            .with_status(200)
            .with_body_from_file(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "update_table_response.json"
            ))
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table1 = {
            let file = File::open(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "create_table_response.json"
            ))
            .unwrap();
            let reader = BufReader::new(file);
            let resp = serde_json::from_reader::<_, LoadTableResult>(reader).unwrap();

            Table::builder()
                .metadata(resp.metadata)
                .metadata_location(resp.metadata_location.unwrap())
                .identifier(TableIdent::from_strs(["ns1", "test1"]).unwrap())
                .file_io(FileIO::new_with_fs())
                .build()
                .unwrap()
        };

        let tx = Transaction::new(&table1);
        let table = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V2)
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        assert_eq!(
            &TableIdent::from_strs(vec!["ns1", "test1"]).unwrap(),
            table.identifier()
        );
        assert_eq!(
            "s3://warehouse/database/table/metadata.json",
            table.metadata_location().unwrap()
        );
        assert_eq!(FormatVersion::V2, table.metadata().format_version());
        assert_eq!("s3://warehouse/database/table", table.metadata().location());
        assert_eq!(
            uuid!("bf289591-dcc0-4234-ad4f-5c3eed811a29"),
            table.metadata().uuid()
        );
        assert_eq!(
            1657810967051,
            table
                .metadata()
                .last_updated_timestamp()
                .unwrap()
                .timestamp_millis()
        );
        assert_eq!(
            vec![&Arc::new(
                Schema::builder()
                    .with_fields(vec![
                        NestedField::optional(1, "foo", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::required(2, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(3, "baz", Type::Primitive(PrimitiveType::Boolean))
                            .into(),
                    ])
                    .with_schema_id(0)
                    .with_identifier_field_ids(vec![2])
                    .build()
                    .unwrap()
            )],
            table.metadata().schemas_iter().collect::<Vec<_>>()
        );
        assert_eq!(
            &HashMap::from([
                (
                    "write.delete.parquet.compression-codec".to_string(),
                    "zstd".to_string()
                ),
                (
                    "write.metadata.compression-codec".to_string(),
                    "gzip".to_string()
                ),
                (
                    "write.summary.partition-limit".to_string(),
                    "100".to_string()
                ),
                (
                    "write.parquet.compression-codec".to_string(),
                    "zstd".to_string()
                ),
            ]),
            table.metadata().properties()
        );
        assert!(table.metadata().current_snapshot().is_none());
        assert!(table.metadata().history().is_empty());
        assert_eq!(
            vec![&Arc::new(SortOrder {
                order_id: 0,
                fields: vec![],
            })],
            table.metadata().sort_orders_iter().collect::<Vec<_>>()
        );

        config_mock.assert_async().await;
        update_table_mock.assert_async().await;
        load_table_mock.assert_async().await
    }

    // A 5xx on the commit POST means the service may have applied the update before failing.
    // Java `ErrorHandlers$CommitErrorHandler` maps it to `CommitStateUnknownException`. A retry
    // would re-apply the same data files and duplicate rows. Pins through the full
    // `Transaction::commit` stack that the unknown-outcome kind survives, is not retryable, and
    // the POST fires exactly once. Discriminates reclassifying 502 as retryable (row R157).
    #[tokio::test]
    async fn test_update_table_502_unknown_outcome_surfaces_without_retry() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let load_table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/test1")
            .with_status(200)
            .with_body_from_file(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "load_table_response.json"
            ))
            .expect_at_least(1) // a (mutated-in) retry re-loads; hits are asserted on the POST
            .create_async()
            .await;

        let update_table_mock = server
            .mock("POST", "/v1/namespaces/ns1/tables/test1")
            .with_status(502)
            .with_body(
                r#"{"error": {"message": "bad gateway", "type": "ServiceFailureException", "code": 502}}"#,
            )
            .expect(1) // the unknown-outcome commit must NOT be re-sent
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table1 = {
            let file = File::open(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "create_table_response.json"
            ))
            .unwrap();
            let reader = BufReader::new(file);
            let resp = serde_json::from_reader::<_, LoadTableResult>(reader).unwrap();

            Table::builder()
                .metadata(resp.metadata)
                .metadata_location(resp.metadata_location.unwrap())
                .identifier(TableIdent::from_strs(["ns1", "test1"]).unwrap())
                .file_io(FileIO::new_with_fs())
                .build()
                .unwrap()
        };

        let tx = Transaction::new(&table1);
        let error = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V2)
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .expect_err("a 502 commit must surface the unknown outcome, not succeed");

        assert_eq!(error.kind(), ErrorKind::CommitStateUnknown);
        assert!(error.to_string().contains("commit state is unknown"));
        assert!(
            !error.retryable(),
            "an unknown-outcome commit error must not advertise retryability"
        );

        config_mock.assert_async().await;
        load_table_mock.assert_async().await;
        update_table_mock.assert_async().await;
    }

    // An HTTP 200 means the commit landed. An unparsable body must not become a generic error,
    // or the caller re-runs a durable commit and duplicates it. Java has no analogue arm; this
    // extends `CommitStateUnknownException` (row R157). Pins through the full
    // `Transaction::commit` stack that the kind is `CommitStateUnknown`, is not retryable, and
    // the POST fires exactly once. Discriminates mutating the OK-arm kind to `Unexpected`.
    #[tokio::test]
    async fn test_update_table_200_unparsable_body_maps_to_commit_state_unknown() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let load_table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/test1")
            .with_status(200)
            .with_body_from_file(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "load_table_response.json"
            ))
            .expect_at_least(1) // a (mutated-in) retry re-loads; hits are asserted on the POST
            .create_async()
            .await;

        // 200 with a body that is NOT a valid CommitTableResponse: the commit landed, but the
        // outcome payload is unreadable (truncated proxy body, wrong content, etc.).
        let update_table_mock = server
            .mock("POST", "/v1/namespaces/ns1/tables/test1")
            .with_status(200)
            .with_body(r#"certainly-not-json{{{"#)
            .expect(1) // the commit landed — it must NOT be re-sent
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table1 = {
            let file = File::open(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "create_table_response.json"
            ))
            .unwrap();
            let reader = BufReader::new(file);
            let resp = serde_json::from_reader::<_, LoadTableResult>(reader).unwrap();

            Table::builder()
                .metadata(resp.metadata)
                .metadata_location(resp.metadata_location.unwrap())
                .identifier(TableIdent::from_strs(["ns1", "test1"]).unwrap())
                .file_io(FileIO::new_with_fs())
                .build()
                .unwrap()
        };

        let tx = Transaction::new(&table1);
        let error = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V2)
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .expect_err("a 200 with an unreadable body must surface the unknown outcome");

        assert_eq!(error.kind(), ErrorKind::CommitStateUnknown);
        assert!(error.to_string().contains("returned HTTP 200"));
        assert!(
            !error.retryable(),
            "a landed-but-unreadable commit must not advertise retryability"
        );

        config_mock.assert_async().await;
        load_table_mock.assert_async().await;
        update_table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_update_table_404() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let load_table_mock = server
            .mock("GET", "/v1/namespaces/ns1/tables/test1")
            .with_status(200)
            .with_body_from_file(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "load_table_response.json"
            ))
            .create_async()
            .await;

        let update_table_mock = server
            .mock("POST", "/v1/namespaces/ns1/tables/test1")
            .with_status(404)
            .with_body(
                r#"
{
    "error": {
        "message": "The given table does not exist",
        "type": "NoSuchTableException",
        "code": 404
    }
}
            "#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table1 = {
            let file = File::open(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "create_table_response.json"
            ))
            .unwrap();
            let reader = BufReader::new(file);
            let resp = serde_json::from_reader::<_, LoadTableResult>(reader).unwrap();

            Table::builder()
                .metadata(resp.metadata)
                .metadata_location(resp.metadata_location.unwrap())
                .identifier(TableIdent::from_strs(["ns1", "test1"]).unwrap())
                .file_io(FileIO::new_with_fs())
                .build()
                .unwrap()
        };

        let tx = Transaction::new(&table1);
        let table_result = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V2)
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await;

        assert!(table_result.is_err());
        assert!(
            table_result
                .err()
                .unwrap()
                .message()
                .contains("does not exist")
        );

        config_mock.assert_async().await;
        update_table_mock.assert_async().await;
        load_table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_register_table() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let register_table_mock = server
            .mock("POST", "/v1/namespaces/ns1/register")
            .with_status(200)
            .with_body_from_file(format!(
                "{}/testdata/{}",
                env!("CARGO_MANIFEST_DIR"),
                "load_table_response.json"
            ))
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );
        let table_ident =
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "test1".to_string());
        let metadata_location = String::from(
            "s3://warehouse/database/table/metadata/00001-5f2f8166-244c-4eae-ac36-384ecdec81fc.gz.metadata.json",
        );

        let table = catalog
            .register_table(&table_ident, metadata_location)
            .await
            .unwrap();

        assert_eq!(
            &TableIdent::from_strs(vec!["ns1", "test1"]).unwrap(),
            table.identifier()
        );
        assert_eq!(
            "s3://warehouse/database/table/metadata/00001-5f2f8166-244c-4eae-ac36-384ecdec81fc.gz.metadata.json",
            table.metadata_location().unwrap()
        );

        config_mock.assert_async().await;
        register_table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_register_table_404() {
        let mut server = Server::new_async().await;

        let config_mock = create_config_mock(&mut server).await;

        let register_table_mock = server
            .mock("POST", "/v1/namespaces/ns1/register")
            .with_status(404)
            .with_body(
                r#"
{
    "error": {
        "message": "The namespace specified does not exist",
        "type": "NoSuchNamespaceErrorException",
        "code": 404
    }
}
            "#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let table_ident =
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "test1".to_string());
        let metadata_location = String::from(
            "s3://warehouse/database/table/metadata/00001-5f2f8166-244c-4eae-ac36-384ecdec81fc.gz.metadata.json",
        );
        let table = catalog
            .register_table(&table_ident, metadata_location)
            .await;

        assert!(table.is_err());
        assert!(table.err().unwrap().message().contains("does not exist"));

        config_mock.assert_async().await;
        register_table_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_create_rest_catalog() {
        let builder = RestCatalogBuilder::default().with_client(Client::new());

        let catalog = builder
            .load(
                "test",
                HashMap::from([
                    (
                        REST_CATALOG_PROP_URI.to_string(),
                        "http://localhost:8080".to_string(),
                    ),
                    ("a".to_string(), "b".to_string()),
                ]),
            )
            .await;

        assert!(catalog.is_ok());

        let catalog_config = catalog.unwrap().user_config;
        assert_eq!(catalog_config.name.as_deref(), Some("test"));
        assert_eq!(catalog_config.uri, "http://localhost:8080");
        assert_eq!(catalog_config.warehouse, None);
        assert!(catalog_config.client.is_some());

        assert_eq!(catalog_config.props.get("a"), Some(&"b".to_string()));
        assert!(!catalog_config.props.contains_key(REST_CATALOG_PROP_URI));
    }

    #[tokio::test]
    async fn test_create_rest_catalog_no_uri() {
        let builder = RestCatalogBuilder::default();

        let catalog = builder
            .load(
                "test",
                HashMap::from([(
                    REST_CATALOG_PROP_WAREHOUSE.to_string(),
                    "s3://warehouse".to_string(),
                )]),
            )
            .await;

        assert!(catalog.is_err());
        if let Err(err) = catalog {
            assert_eq!(err.kind(), ErrorKind::DataInvalid);
            assert_eq!(err.message(), "Catalog uri is required");
        }
    }

    // ========================================================================
    // View method wiring tests — confirm each view method targets the correct REST route and maps
    // status codes to the right outcome. (Shape-level: a mock server, no real catalog backend.)
    // ========================================================================

    fn view_metadata_body() -> &'static str {
        r#"{
            "metadata-location": "s3://iceberg-catalog/ns1/view1/metadata/00001-abc.metadata.json",
            "metadata": {
                "view-uuid": "fa6506c3-7681-40c8-86dc-e36561f83385",
                "format-version": 1,
                "location": "s3://iceberg-catalog/ns1/view1",
                "current-version-id": 1,
                "properties": {},
                "versions": [ {
                    "version-id": 1,
                    "timestamp-ms": 1573518431292,
                    "schema-id": 1,
                    "default-namespace": [ "ns1" ],
                    "summary": {},
                    "representations": [ {
                        "type": "sql",
                        "sql": "SELECT 1 AS event_count",
                        "dialect": "spark"
                    } ]
                } ],
                "schemas": [ {
                    "schema-id": 1,
                    "type": "struct",
                    "fields": [ {
                        "id": 1,
                        "name": "event_count",
                        "required": false,
                        "type": "int"
                    } ]
                } ],
                "version-log": [ {
                    "timestamp-ms": 1573518431292,
                    "version-id": 1
                } ]
            },
            "config": {}
        }"#
    }

    // RISK: list_views must hit `GET /namespaces/{ns}/views` (NOT /tables) and parse the shared
    // identifiers response shape.
    #[tokio::test]
    async fn test_list_views_targets_views_route() {
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;
        let list_views_mock = server
            .mock("GET", "/v1/namespaces/ns1/views")
            .with_status(200)
            .with_body(
                r#"{
                "identifiers": [
                    { "namespace": ["ns1"], "name": "view1" },
                    { "namespace": ["ns1"], "name": "view2" }
                ]
            }"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let views = catalog
            .list_views(&NamespaceIdent::new("ns1".to_string()))
            .await
            .unwrap();
        assert_eq!(views, vec![
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "view1".to_string()),
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "view2".to_string()),
        ]);

        config_mock.assert_async().await;
        list_views_mock.assert_async().await;
    }

    // RISK: load_view must hit `GET /namespaces/{ns}/views/{view}` and decode the full
    // LoadViewResult (metadata-location + the ViewMetadata).
    #[tokio::test]
    async fn test_load_view_decodes_metadata() {
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;
        let load_view_mock = server
            .mock("GET", "/v1/namespaces/ns1/views/view1")
            .with_status(200)
            .with_body(view_metadata_body())
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let view_ident =
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "view1".to_string());
        let view = catalog.load_view(&view_ident).await.unwrap();
        assert_eq!(view.identifier(), &view_ident);
        assert_eq!(
            view.metadata().uuid().to_string(),
            "fa6506c3-7681-40c8-86dc-e36561f83385"
        );
        assert_eq!(view.metadata().current_version_id(), 1);
        assert_eq!(
            view.metadata_location(),
            Some("s3://iceberg-catalog/ns1/view1/metadata/00001-abc.metadata.json")
        );

        config_mock.assert_async().await;
        load_view_mock.assert_async().await;
    }

    // RISK: load_view on a 404 must map to ViewNotFound (not a generic error), so callers can
    // distinguish "absent" from "transport failure".
    #[tokio::test]
    async fn test_load_view_not_found_maps_to_view_not_found() {
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;
        let load_view_mock = server
            .mock("GET", "/v1/namespaces/ns1/views/absent")
            .with_status(404)
            .with_body(
                r#"{"error": {"message": "View does not exist", "type": "NoSuchViewException", "code": 404}}"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let error = catalog
            .load_view(&TableIdent::new(
                NamespaceIdent::new("ns1".to_string()),
                "absent".to_string(),
            ))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ViewNotFound);

        config_mock.assert_async().await;
        load_view_mock.assert_async().await;
    }

    // RISK: view_exists must HEAD the view route and map 204→true, 404→false.
    #[tokio::test]
    async fn test_view_exists_maps_head_status() {
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;
        let exists_mock = server
            .mock("HEAD", "/v1/namespaces/ns1/views/view1")
            .with_status(204)
            .create_async()
            .await;
        let absent_mock = server
            .mock("HEAD", "/v1/namespaces/ns1/views/absent")
            .with_status(404)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        assert!(
            catalog
                .view_exists(&TableIdent::new(
                    NamespaceIdent::new("ns1".to_string()),
                    "view1".to_string()
                ))
                .await
                .unwrap()
        );
        assert!(
            !catalog
                .view_exists(&TableIdent::new(
                    NamespaceIdent::new("ns1".to_string()),
                    "absent".to_string()
                ))
                .await
                .unwrap()
        );

        config_mock.assert_async().await;
        exists_mock.assert_async().await;
        absent_mock.assert_async().await;
    }

    // RISK: drop_view must DELETE the view route and treat 404 as ViewNotFound (not a no-op).
    #[tokio::test]
    async fn test_drop_view_targets_view_route_and_maps_not_found() {
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;
        let drop_mock = server
            .mock("DELETE", "/v1/namespaces/ns1/views/view1")
            .with_status(204)
            .create_async()
            .await;
        let drop_absent_mock = server
            .mock("DELETE", "/v1/namespaces/ns1/views/absent")
            .with_status(404)
            .with_body(
                r#"{"error": {"message": "View does not exist", "type": "NoSuchViewException", "code": 404}}"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        catalog
            .drop_view(&TableIdent::new(
                NamespaceIdent::new("ns1".to_string()),
                "view1".to_string(),
            ))
            .await
            .unwrap();
        let error = catalog
            .drop_view(&TableIdent::new(
                NamespaceIdent::new("ns1".to_string()),
                "absent".to_string(),
            ))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ViewNotFound);

        config_mock.assert_async().await;
        drop_mock.assert_async().await;
        drop_absent_mock.assert_async().await;
    }

    // RISK: rename_view must POST `/views/rename` (NOT /tables/rename) and map 409 to
    // ViewAlreadyExists.
    #[tokio::test]
    async fn test_rename_view_targets_views_rename_route() {
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;
        let rename_mock = server
            .mock("POST", "/v1/views/rename")
            .with_status(204)
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        catalog
            .rename_view(
                &TableIdent::new(NamespaceIdent::new("ns1".to_string()), "v_src".to_string()),
                &TableIdent::new(NamespaceIdent::new("ns1".to_string()), "v_dst".to_string()),
            )
            .await
            .unwrap();

        config_mock.assert_async().await;
        rename_mock.assert_async().await;
    }

    // RISK: update_view must POST the view route with the commit body and map 409 to a retryable
    // CatalogCommitConflicts (the REST server does the CAS; the client surfaces the conflict).
    #[tokio::test]
    async fn test_update_view_maps_conflict_to_retryable() {
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;
        let load_view_mock = server
            .mock("GET", "/v1/namespaces/ns1/views/view1")
            .with_status(200)
            .with_body(view_metadata_body())
            .create_async()
            .await;
        // Pin the commit body, not just the route. A table requirement or update tag leaking
        // into the view request must fail to match this mock.
        let commit_mock = server
            .mock("POST", "/v1/namespaces/ns1/views/view1")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(r#""identifier""#.to_string()),
                mockito::Matcher::Regex(r#""name":"view1""#.to_string()),
                mockito::Matcher::Regex(r#""type":"assert-view-uuid""#.to_string()),
                mockito::Matcher::Regex(
                    "fa6506c3-7681-40c8-86dc-e36561f83385".to_string(),
                ),
                mockito::Matcher::Regex(r#""action":"set-properties""#.to_string()),
                mockito::Matcher::Regex(r#""comment":"daily""#.to_string()),
            ]))
            .with_status(409)
            .with_body(
                r#"{"error": {"message": "metadata location has changed", "type": "CommitFailedException", "code": 409}}"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let view_ident =
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "view1".to_string());
        let view = catalog.load_view(&view_ident).await.unwrap();
        let commit = view
            .update_properties()
            .set("comment", "daily")
            .unwrap()
            .to_commit()
            .unwrap();
        let error = catalog.update_view(commit).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::CatalogCommitConflicts);
        assert!(error.retryable());

        config_mock.assert_async().await;
        load_view_mock.assert_async().await;
        commit_mock.assert_async().await;
    }

    // A 5xx view commit must surface `CommitStateUnknown`, not the generic transport arm, and
    // must not be retryable. Re-issuing a possibly-applied commit is unsafe.
    #[tokio::test]
    async fn test_update_view_5xx_maps_to_commit_state_unknown() {
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;
        let load_view_mock = server
            .mock("GET", "/v1/namespaces/ns1/views/view1")
            .with_status(200)
            .with_body(view_metadata_body())
            .create_async()
            .await;
        // 503 Service Unavailable: the commit may or may not have landed.
        let commit_mock = server
            .mock("POST", "/v1/namespaces/ns1/views/view1")
            .with_status(503)
            .with_body(
                r#"{"error": {"message": "service unavailable", "type": "ServiceUnavailableException", "code": 503}}"#,
            )
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let view_ident =
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "view1".to_string());
        let view = catalog.load_view(&view_ident).await.unwrap();
        let commit = view
            .update_properties()
            .set("comment", "daily")
            .unwrap()
            .to_commit()
            .unwrap();
        let error = catalog.update_view(commit).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::CommitStateUnknown);
        assert!(error.to_string().contains("commit state is unknown"));
        // A state-unknown commit must NOT be auto-retried.
        assert!(!error.retryable());

        config_mock.assert_async().await;
        load_view_mock.assert_async().await;
        commit_mock.assert_async().await;
    }

    // A view commit returning 200 landed, so an unparsable body must not read as retryable
    // (row R157). Pins kind, retryability, and one POST. Discriminates mutating the OK-arm kind.
    #[tokio::test]
    async fn test_update_view_200_unparsable_body_maps_to_commit_state_unknown() {
        let mut server = Server::new_async().await;
        let config_mock = create_config_mock(&mut server).await;
        let load_view_mock = server
            .mock("GET", "/v1/namespaces/ns1/views/view1")
            .with_status(200)
            .with_body(view_metadata_body())
            .create_async()
            .await;
        // 200 with a body that is NOT a valid LoadViewResult: the commit landed, but the
        // outcome payload is unreadable.
        let commit_mock = server
            .mock("POST", "/v1/namespaces/ns1/views/view1")
            .with_status(200)
            .with_body(r#"certainly-not-json{{{"#)
            .expect(1) // the commit landed — it must NOT be re-sent
            .create_async()
            .await;

        let catalog = RestCatalog::new(
            RestCatalogConfig::builder().uri(server.url()).build(),
            Some(Arc::new(LocalFsStorageFactory)),
        );

        let view_ident =
            TableIdent::new(NamespaceIdent::new("ns1".to_string()), "view1".to_string());
        let view = catalog.load_view(&view_ident).await.unwrap();
        let commit = view
            .update_properties()
            .set("comment", "daily")
            .unwrap()
            .to_commit()
            .unwrap();
        let error = catalog.update_view(commit).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::CommitStateUnknown);
        assert!(error.to_string().contains("returned HTTP 200"));
        assert!(
            !error.retryable(),
            "a landed-but-unreadable view commit must not advertise retryability"
        );

        config_mock.assert_async().await;
        load_view_mock.assert_async().await;
        commit_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_name_and_properties_return_user_config() {
        // The accessors read `user_config` and need no network round-trip.
        let config = RestCatalogConfig::builder()
            .name("rest_cat".to_string())
            .uri("http://localhost:8181".to_string())
            .props(HashMap::from([("k".to_string(), "v".to_string())]))
            .build();
        let catalog = RestCatalog::new(config, Some(Arc::new(LocalFsStorageFactory)));

        assert_eq!(catalog.name(), "rest_cat");
        // Mutation guard: the empty-map default fails here.
        assert_eq!(catalog.properties().get("k").map(String::as_str), Some("v"));
    }

    #[tokio::test]
    async fn test_name_defaults_to_sentinel_when_unset() {
        // No `name` was set on the config, so the override falls back to the sentinel.
        let config = RestCatalogConfig::builder()
            .uri("http://localhost:8181".to_string())
            .build();
        let catalog = RestCatalog::new(config, Some(Arc::new(LocalFsStorageFactory)));
        assert_eq!(catalog.name(), UNNAMED_CATALOG);
    }

    #[tokio::test]
    async fn test_invalidate_defaults_are_noops() {
        let config = RestCatalogConfig::builder()
            .uri("http://localhost:8181".to_string())
            .build();
        let catalog = RestCatalog::new(config, Some(Arc::new(LocalFsStorageFactory)));
        let ident = TableIdent::new(NamespaceIdent::new("ns".to_string()), "t".to_string());
        // No network: the inherited no-op defaults return Ok without touching the server.
        catalog.invalidate_table(&ident).await.unwrap();
        catalog.invalidate_view(&ident).await.unwrap();
    }
}
