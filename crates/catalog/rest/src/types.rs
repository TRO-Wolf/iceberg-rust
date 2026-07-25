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

//! Request and response types for the Iceberg REST API.

use std::collections::HashMap;

use iceberg::io::is_secret_prop_key;
use iceberg::spec::{
    Schema, SortOrder, TableMetadata, UnboundPartitionSpec, ViewMetadata, ViewVersion,
};
use iceberg::{
    Error, ErrorKind, Namespace, NamespaceIdent, TableIdent, TableRequirement, TableUpdate,
    ViewRequirement, ViewUpdate,
};
use serde_derive::{Deserialize, Serialize};

// ============================================================================
// Secret redaction for `Debug` (SEC-010)
//
// Several wire types in this module carry live credential material: the OAuth
// `access_token`, the server-vended storage credentials, and the `String -> String`
// property/config maps the REST server returns — which the catalog copies straight into
// the table's `FileIO` props (see `catalog.rs::load_file_io`). A plain `#[derive(Debug)]`
// prints all of that in clear at any `{:?}` / `tracing` site, including sites in
// downstream user code this crate does not control, so every secret-bearing type below
// carries a hand-written `Debug` instead.
//
// Property maps are redacted PER KEY through the canonical needle test
// `iceberg::io::is_secret_prop_key` (`crates/iceberg/src/io/storage/config/mod.rs`) — the
// same superset `StorageConfig` and the Glue / HMS / S3Tables / SQL config `Debug` impls
// use — so keys stay visible for diagnostics, only secret VALUES are masked, and there is
// exactly ONE authoritative needle list to keep current instead of drifting copies. The
// narrower REST-local exact-match list would miss precisely the vended FileIO credentials
// this module receives (`s3.secret-access-key`, `s3.session-token`, `gcs.oauth2.token`,
// `adls.connection-string`). The needle test is deliberately a SUPERSET: a non-secret key
// whose name merely contains a needle (e.g. `token-refresh-enabled`) renders as `***` too.
// Over-redaction is the safe direction for a debug view.
//
// ASSESSED and deliberately left on `#[derive(Debug)]` — no `String -> String` value map
// and no secret field of their own:
//   * Identifier / pagination-only shapes: `ListNamespaceResponse`, `ListTablesResponse`,
//     `RenameTableRequest`, `RegisterTableRequest`, `UpdateNamespacePropertiesResponse`.
//
// ASSESSED as RESIDUE, not as an all-clear — left on `#[derive(Debug)]` because redacting
// only `Debug` would be theater while `Display` carries the same values:
//   * `ErrorModel` / `ErrorResponse` — `message` / `type` / `code` / `stack` are the
//     server's diagnostic payload, and `From<ErrorModel> for Error` already surfaces them
//     verbatim. They are server-controlled free text: a hostile or careless server can echo
//     whatever it likes into `message`, and this is now the one channel by which content from
//     a token-endpoint / catalog response body still reaches logs, since
//     `deserialize_catalog_response` stopped attaching raw bodies (SEC-010/F1).
//   * `OAuthError` — RFC 6749 §5.2 fields (`error` is a fixed error code, `error_description`
//     is human-readable text, `error_uri` a documentation link), all three already surfaced by
//     `From<OAuthError> for Error`. `error_description` is likewise server-controlled free
//     text, so the same residue applies: the SHAPE carries no client secret, but the CONTENT
//     is the server's to choose.
//
// NAMED RESIDUE (core crate, out of scope for this unit) — `String -> String` maps that still
// derive `Debug` in `crates/iceberg` and print in clear:
//   * `TableMetadata.properties` — reachable via `LoadTableResult.metadata` and
//     `CommitTableResponse.metadata` (which is why `CommitTableResponse` is NOT in the
//     all-clear list above: it carries a full `TableMetadata`).
//   * `ViewMetadata.properties` — reachable via `LoadViewResult.metadata`.
//   * `ViewVersion.summary` — reachable via `CreateViewRequest.view_version` and, nested,
//     through both view-metadata paths.
//   * `TableUpdate::SetProperties` / `ViewUpdate::SetProperties` — reachable via
//     `CommitTableRequest.updates` / `CommitViewRequest.updates`.
// So a `{:?}` of those fields can still surface a credential an operator stored as a TABLE or
// VIEW property. Closing the core-crate property maps is a separate unit.
//
// SCOPE OF THIS FIX — the REST server's own credential channels (`config`,
// `storage-credentials`) are covered AT THE `Debug` LAYER, which is not the same as "covered".
// A credential-bearing body that fails to PARSE can still reach logs through the
// `serde_json` error attached as the `source` of the parse error, because `iceberg::Error`
// renders its source verbatim; the echo is unbounded when the type mismatch sits at a
// container boundary (double-encoded JSON). That path is documented on
// `client.rs::deserialize_catalog_response` and pinned as known residue by
// `test_known_residue_double_encoded_body_leaks_through_error_source`.
// ============================================================================

/// Marker written in place of a redacted secret value. Its presence also signals that the
/// field/entry was populated, which is the only thing a debug view legitimately needs.
const REDACTED: &str = "***";

/// `Debug` adapter that renders a property map with secret-bearing VALUES masked.
///
/// Keys are always printed — they are diagnostic, not secret. A value whose key satisfies
/// [`is_secret_prop_key`] is replaced by [`REDACTED`]. See the module's redaction banner for
/// why the canonical (superset) needle test is used rather than a REST-local list.
struct RedactedProps<'a>(&'a HashMap<String, String>);

impl std::fmt::Debug for RedactedProps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.0.iter().map(|(k, v)| {
                let value = if is_secret_prop_key(k) {
                    REDACTED
                } else {
                    v.as_str()
                };
                (k.as_str(), value)
            }))
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct CatalogConfig {
    pub(super) overrides: HashMap<String, String>,
    pub(super) defaults: HashMap<String, String>,
}

impl std::fmt::Debug for CatalogConfig {
    /// Hand-written: the `GET /v1/config` response's `defaults`/`overrides` are merged into the
    /// catalog's runtime props and handed to `FileIO`, so they routinely carry vended storage
    /// credentials (`s3.secret-access-key`, `s3.session-token`, …) as well as `token`. Secret
    /// VALUES are masked per key; keys and non-secret values stay readable.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatalogConfig")
            .field("overrides", &RedactedProps(&self.overrides))
            .field("defaults", &RedactedProps(&self.defaults))
            .finish()
    }
}

#[derive(Debug, Serialize, Deserialize)]
/// Wrapper for all non-2xx error responses from the REST API
pub struct ErrorResponse {
    error: ErrorModel,
}

impl From<ErrorResponse> for Error {
    fn from(resp: ErrorResponse) -> Error {
        resp.error.into()
    }
}

#[derive(Debug, Serialize, Deserialize)]
/// Error payload returned in a response with further details on the error
pub struct ErrorModel {
    /// Human-readable error message
    pub message: String,
    /// Internal type definition of the error
    pub r#type: String,
    /// HTTP response code
    pub code: u16,
    /// Optional error stack / context
    pub stack: Option<Vec<String>>,
}

impl From<ErrorModel> for Error {
    fn from(value: ErrorModel) -> Self {
        let mut error = Error::new(ErrorKind::DataInvalid, value.message)
            .with_context("type", value.r#type)
            .with_context("code", format!("{}", value.code));

        if let Some(stack) = value.stack {
            error = error.with_context("stack", stack.join("\n"));
        }

        error
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OAuthError {
    pub(super) error: String,
    pub(super) error_description: Option<String>,
    pub(super) error_uri: Option<String>,
}

impl From<OAuthError> for Error {
    fn from(value: OAuthError) -> Self {
        let mut error = Error::new(
            ErrorKind::DataInvalid,
            format!("OAuthError: {}", value.error),
        );

        if let Some(desc) = value.error_description {
            error = error.with_context("description", desc);
        }

        if let Some(uri) = value.error_uri {
            error = error.with_context("uri", uri);
        }

        error
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct TokenResponse {
    pub(super) access_token: String,
    pub(super) token_type: String,
    pub(super) expires_in: Option<u64>,
    pub(super) issued_token_type: Option<String>,
}

impl std::fmt::Debug for TokenResponse {
    /// Hand-written: `access_token` is raw bearer-secret material straight off the OAuth token
    /// endpoint. A derived `Debug` would print it at any `{:?}` / `tracing::error!(?resp)` site.
    /// The remaining fields (`token_type`, `expires_in`, `issued_token_type`) are non-secret OAuth
    /// metadata and stay readable so a failing exchange is still diagnosable.
    ///
    /// Only presence is signalled (via [`REDACTED`]); the token's LENGTH is deliberately not
    /// emitted — a length is a weak oracle on secret material and buys nothing here.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenResponse")
            .field("access_token", &REDACTED)
            .field("token_type", &self.token_type)
            .field("expires_in", &self.expires_in)
            .field("issued_token_type", &self.issued_token_type)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Namespace response
pub struct NamespaceResponse {
    /// Namespace identifier
    pub namespace: NamespaceIdent,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    /// Properties stored on the namespace, if supported by the server.
    pub properties: HashMap<String, String>,
}

impl std::fmt::Debug for NamespaceResponse {
    /// Hand-written: `properties` is a server-returned `String -> String` map whose entries are
    /// redacted per key (see the module's redaction banner).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamespaceResponse")
            .field("namespace", &self.namespace)
            .field("properties", &RedactedProps(&self.properties))
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Create namespace request
pub struct CreateNamespaceRequest {
    /// Name of the namespace to create
    pub namespace: NamespaceIdent,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    /// Properties to set on the namespace
    pub properties: HashMap<String, String>,
}

impl std::fmt::Debug for CreateNamespaceRequest {
    /// Hand-written: `properties` may carry credentials an operator set on the namespace; entries
    /// are redacted per key (see the module's redaction banner).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateNamespaceRequest")
            .field("namespace", &self.namespace)
            .field("properties", &RedactedProps(&self.properties))
            .finish()
    }
}

impl From<&Namespace> for NamespaceResponse {
    fn from(value: &Namespace) -> Self {
        Self {
            namespace: value.name().clone(),
            properties: value.properties().clone(),
        }
    }
}

impl From<NamespaceResponse> for Namespace {
    fn from(value: NamespaceResponse) -> Self {
        Namespace::with_properties(value.namespace, value.properties)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Response containing a list of namespace identifiers, with optional pagination support.
pub struct ListNamespaceResponse {
    /// List of namespace identifiers returned by the server
    pub namespaces: Vec<NamespaceIdent>,
    /// Opaque token for pagination. If present, indicates there are more results available.
    /// Use this value in subsequent requests to retrieve the next page.
    pub next_page_token: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Request to update properties on a namespace.
///
/// Properties that are not in the request are not modified or removed by this call.
/// Server implementations are not required to support namespace properties.
pub struct UpdateNamespacePropertiesRequest {
    /// List of property keys to remove from the namespace
    pub removals: Option<Vec<String>>,
    /// Map of property keys to values to set or update on the namespace
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub updates: HashMap<String, String>,
}

impl std::fmt::Debug for UpdateNamespacePropertiesRequest {
    /// Hand-written: `updates` carries the property VALUES being written, which may include
    /// credentials; entries are redacted per key. `removals` is key-only and stays readable.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpdateNamespacePropertiesRequest")
            .field("removals", &self.removals)
            .field("updates", &RedactedProps(&self.updates))
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Response from updating namespace properties, indicating which properties were changed.
pub struct UpdateNamespacePropertiesResponse {
    /// List of property keys that were added or updated
    pub updated: Vec<String>,
    /// List of properties that were removed
    pub removed: Vec<String>,
    /// List of properties requested for removal that were not found in the namespace's properties.
    /// Represents a partial success response. Servers do not need to implement this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub missing: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Response containing a list of table identifiers, with optional pagination support.
pub struct ListTablesResponse {
    /// List of table identifiers under the requested namespace
    pub identifiers: Vec<TableIdent>,
    /// Opaque token for pagination. If present, indicates there are more results available.
    /// Use this value in subsequent requests to retrieve the next page.
    #[serde(default)]
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Request to rename a table from one identifier to another.
///
/// It's valid to move a table across namespaces, but the server implementation
/// is not required to support it.
pub struct RenameTableRequest {
    /// Current table identifier to rename
    pub source: TableIdent,
    /// New table identifier to rename to
    pub destination: TableIdent,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Result returned when a table is successfully loaded or created.
///
/// The table metadata JSON is returned in the `metadata` field. The corresponding file location
/// of table metadata should be returned in the `metadata_location` field, unless the metadata
/// is not yet committed. For example, a create transaction may return metadata that is staged
/// but not committed.
///
/// The `config` map returns table-specific configuration for the table's resources, including
/// its HTTP client and FileIO. For example, config may contain a specific FileIO implementation
/// class for the table depending on its underlying storage.
pub struct LoadTableResult {
    /// May be null if the table is staged as part of a transaction
    pub metadata_location: Option<String>,
    /// The table's full metadata
    pub metadata: TableMetadata,
    /// Table-specific configuration overriding catalog configuration
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub config: HashMap<String, String>,
    /// Storage credentials for accessing table data. Clients should check this field
    /// before falling back to credentials in the `config` field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_credentials: Option<Vec<StorageCredential>>,
}

impl std::fmt::Debug for LoadTableResult {
    /// Hand-written: `config` is documented by the REST spec as a place servers vend table
    /// credentials, and `storage_credentials` is credential material by definition. `config`
    /// entries are redacted per key; `storage_credentials` renders through [`StorageCredential`]'s
    /// own redacting `Debug`.
    ///
    /// RESIDUE: `metadata` still renders through core's derived `TableMetadata` `Debug`, which
    /// prints `properties` in clear — see the module's redaction banner.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadTableResult")
            .field("metadata_location", &self.metadata_location)
            .field("metadata", &self.metadata)
            .field("config", &RedactedProps(&self.config))
            .field("storage_credentials", &self.storage_credentials)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Storage credential for a specific location prefix.
///
/// Indicates a storage location prefix where the credential is relevant. Clients should
/// choose the most specific prefix (by selecting the longest prefix) if several credentials
/// of the same type are available.
pub struct StorageCredential {
    /// Storage location prefix where this credential is relevant
    pub prefix: String,
    /// Configuration map containing credential information
    pub config: HashMap<String, String>,
}

impl std::fmt::Debug for StorageCredential {
    /// Hand-written: `config` holds the vended credential itself (`s3.access-key-id`,
    /// `s3.secret-access-key`, `s3.session-token`, …). Values are redacted per key; `prefix` is a
    /// storage location, not a secret, and stays readable so credential selection is diagnosable.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageCredential")
            .field("prefix", &self.prefix)
            .field("config", &RedactedProps(&self.config))
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Request to create a new table in a namespace.
///
/// If `stage_create` is false, the table is created immediately.
/// If `stage_create` is true, the table is not created, but table metadata is initialized
/// and returned. The service should prepare as needed for a commit to the table commit
/// endpoint to complete the create transaction.
pub struct CreateTableRequest {
    /// Name of the table to create
    pub name: String,
    /// Optional table location. If not provided, the server will choose a location.
    pub location: Option<String>,
    /// Table schema
    pub schema: Schema,
    /// Optional partition specification. If not provided, the table will be unpartitioned.
    pub partition_spec: Option<UnboundPartitionSpec>,
    /// Optional sort order for the table
    pub write_order: Option<SortOrder>,
    /// Whether to stage the create for a transaction (true) or create immediately (false)
    pub stage_create: Option<bool>,
    /// Optional properties to set on the table
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub properties: HashMap<String, String>,
}

impl std::fmt::Debug for CreateTableRequest {
    /// Hand-written: `properties` carries the table properties being written, which may include
    /// FileIO credentials; entries are redacted per key. Everything else is schema/layout metadata
    /// and stays readable.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateTableRequest")
            .field("name", &self.name)
            .field("location", &self.location)
            .field("schema", &self.schema)
            .field("partition_spec", &self.partition_spec)
            .field("write_order", &self.write_order)
            .field("stage_create", &self.stage_create)
            .field("properties", &RedactedProps(&self.properties))
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Request to commit updates to a table.
///
/// Commits have two parts: requirements and updates. Requirements are assertions that will
/// be validated before attempting to make and commit changes. Updates are changes to make
/// to table metadata.
///
/// Create table transactions that are started by createTable with `stage-create` set to true
/// are committed using this request. Transactions should include all changes to the table,
/// including table initialization, like AddSchemaUpdate and SetCurrentSchemaUpdate.
pub struct CommitTableRequest {
    /// Table identifier to update; must be present for CommitTransactionRequest
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identifier: Option<TableIdent>,
    /// List of requirements that must be satisfied before committing changes
    pub requirements: Vec<TableRequirement>,
    /// List of updates to apply to the table metadata
    pub updates: Vec<TableUpdate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Response returned when a table is successfully updated.
///
/// The table metadata JSON is returned in the metadata field. The corresponding file location
/// of table metadata must be returned in the metadata-location field. Clients can check whether
/// metadata has changed by comparing metadata locations.
pub struct CommitTableResponse {
    /// Location of the updated table metadata file
    pub metadata_location: String,
    /// The table's updated metadata
    pub metadata: TableMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Request to register a table using an existing metadata file location.
pub struct RegisterTableRequest {
    /// Name of the table to register
    pub name: String,
    /// Location of the metadata file for the table
    pub metadata_location: String,
    /// Whether to overwrite table metadata if the table already exists
    pub overwrite: Option<bool>,
}

// ============================================================================
// View request/response shapes — mirror the Iceberg REST OpenAPI view routes
// (`POST/GET /namespaces/{ns}/views`, `GET/POST/DELETE/HEAD .../views/{view}`,
// `POST /views/rename`) and Java's `CreateViewRequest` / `LoadViewResponse` /
// `UpdateTableRequest` (reused for the view replace/commit) wire formats.
// ============================================================================

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Request to create a new view in a namespace.
///
/// Mirrors Java `org.apache.iceberg.rest.requests.CreateViewRequest` — the wire fields are
/// `name`, `location`, `view-version` (the initial [`ViewVersion`]), `schema`, and `properties`.
pub struct CreateViewRequest {
    /// Name of the view to create
    pub name: String,
    /// Optional view location. If not provided, the server will choose a location.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// The initial view version (one representation per SQL dialect, schema id, default namespace).
    pub view_version: ViewVersion,
    /// View schema
    pub schema: Schema,
    /// Optional properties to set on the view
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub properties: HashMap<String, String>,
}

impl std::fmt::Debug for CreateViewRequest {
    /// Hand-written: `properties` carries the view properties being written, which may include
    /// FileIO credentials; entries are redacted per key.
    ///
    /// RESIDUE: `view_version` renders through core's derived `ViewVersion` `Debug`, whose
    /// `summary` is an unredacted `String -> String` map — see the module's redaction banner.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateViewRequest")
            .field("name", &self.name)
            .field("location", &self.location)
            .field("view_version", &self.view_version)
            .field("schema", &self.schema)
            .field("properties", &RedactedProps(&self.properties))
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Result returned when a view is successfully loaded or created.
///
/// Mirrors Java `org.apache.iceberg.rest.responses.LoadViewResponse` — `metadata-location`,
/// `metadata` (the [`ViewMetadata`] JSON), and a view-specific `config` map.
pub struct LoadViewResult {
    /// Location of the view metadata file
    pub metadata_location: String,
    /// The view's full metadata
    pub metadata: ViewMetadata,
    /// View-specific configuration overriding catalog configuration
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub config: HashMap<String, String>,
}

impl std::fmt::Debug for LoadViewResult {
    /// Hand-written: `config` is the view's server-vended configuration overlay — the same channel
    /// `LoadTableResult.config` uses to hand back credentials — and is redacted per key.
    ///
    /// RESIDUE: `metadata` still renders through core's derived `ViewMetadata` `Debug`, which
    /// prints `properties` in clear — see the module's redaction banner.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadViewResult")
            .field("metadata_location", &self.metadata_location)
            .field("metadata", &self.metadata)
            .field("config", &RedactedProps(&self.config))
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
/// Request to commit updates to a view (the replace/update-properties path).
///
/// Java reuses `UpdateTableRequest.create(identifier, requirements, updates)` for the view commit,
/// so the wire shape is `identifier`, `requirements`, `updates` — but carrying VIEW requirements
/// and VIEW updates (from `UpdateRequirements.forReplaceView`), not table ones.
pub struct CommitViewRequest {
    /// View identifier to update
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identifier: Option<TableIdent>,
    /// List of requirements that must be satisfied before committing changes
    pub requirements: Vec<ViewRequirement>,
    /// List of updates to apply to the view metadata
    pub updates: Vec<ViewUpdate>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // SEC-010 — `Debug` must never render secret VALUES
    //
    // Every test below feeds an obviously-synthetic sentinel into a secret-bearing field and
    // asserts the sentinel is absent from `{:?}`, PLUS asserts that a non-secret sibling field
    // still renders (anti-over-redaction — a `Debug` that prints nothing would pass a
    // leak-only assertion vacuously). Each is RED-able by reverting the type to
    // `#[derive(Debug)]`.
    // ========================================================================

    /// Sentinel standing in for secret material. Deliberately not shaped like a real credential
    /// so it cannot trip a secret scanner.
    const SECRET_SENTINEL: &str = "SENTINEL_MUST_NOT_APPEAR_IN_DEBUG";

    /// Minimal but complete v2 table metadata, so `LoadTableResult` can be built without a
    /// fixture file.
    fn minimal_table_metadata() -> TableMetadata {
        serde_json::from_value(serde_json::json!({
            "format-version": 2,
            "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
            "location": "s3://bucket/warehouse/default.db/t",
            "last-sequence-number": 1,
            "last-updated-ms": 1602638573590i64,
            "last-column-id": 1,
            "current-schema-id": 0,
            "schemas": [ {
                "type": "struct",
                "schema-id": 0,
                "fields": [ { "id": 1, "name": "x", "required": true, "type": "long" } ]
            } ],
            "default-spec-id": 0,
            "partition-specs": [ { "spec-id": 0, "fields": [] } ],
            "last-partition-id": 999,
            "default-sort-order-id": 0,
            "sort-orders": [ { "order-id": 0, "fields": [] } ],
            "properties": {},
            "current-snapshot-id": -1,
            "snapshots": [],
            "snapshot-log": [],
            "metadata-log": []
        }))
        .expect("minimal v2 table metadata must parse")
    }

    fn minimal_schema() -> Schema {
        serde_json::from_value(serde_json::json!({
            "schema-id": 0,
            "type": "struct",
            "fields": [ { "id": 1, "name": "x", "required": true, "type": "long" } ]
        }))
        .expect("minimal schema must parse")
    }

    /// RISK: `TokenResponse.access_token` is raw bearer-secret material off the OAuth token
    /// endpoint. A derived `Debug` prints it at every `{:?}` / `tracing` site.
    /// RED-able: restore `#[derive(Debug)]` on `TokenResponse`.
    #[test]
    fn test_token_response_debug_redacts_access_token() {
        let resp = TokenResponse {
            access_token: SECRET_SENTINEL.to_string(),
            token_type: "bearer".to_string(),
            expires_in: Some(3600),
            issued_token_type: Some("urn:ietf:params:oauth:token-type:access_token".to_string()),
        };

        let debug = format!("{resp:?}");

        assert!(
            !debug.contains(SECRET_SENTINEL),
            "Debug leaked the access token: {debug}"
        );
        assert!(
            debug.contains(REDACTED),
            "expected redaction marker: {debug}"
        );
        // Anti-over-redaction: the non-secret OAuth metadata must stay diagnosable.
        assert!(
            debug.contains("bearer"),
            "Debug dropped token_type: {debug}"
        );
        assert!(debug.contains("3600"), "Debug dropped expires_in: {debug}");
    }

    /// RISK: the `GET /v1/config` `defaults`/`overrides` are merged into the runtime props and
    /// handed to `FileIO`, so they carry vended storage credentials.
    /// RED-able: restore `#[derive(Debug)]` on `CatalogConfig`.
    #[test]
    fn test_catalog_config_debug_redacts_vended_credentials() {
        let config = CatalogConfig {
            overrides: HashMap::from([
                (
                    "s3.secret-access-key".to_string(),
                    SECRET_SENTINEL.to_string(),
                ),
                (
                    "s3.endpoint".to_string(),
                    "https://s3.example.test".to_string(),
                ),
            ]),
            defaults: HashMap::from([
                ("token".to_string(), SECRET_SENTINEL.to_string()),
                ("prefix".to_string(), "ws".to_string()),
            ]),
        };

        let debug = format!("{config:?}");

        assert!(
            !debug.contains(SECRET_SENTINEL),
            "Debug leaked a vended credential: {debug}"
        );
        assert!(
            debug.contains(REDACTED),
            "expected redaction marker: {debug}"
        );
        // Anti-over-redaction: keys stay visible, and non-secret values still render.
        assert!(
            debug.contains("s3.secret-access-key"),
            "Debug dropped the secret KEY (keys are diagnostic, not secret): {debug}"
        );
        assert!(
            debug.contains("https://s3.example.test"),
            "Debug dropped the non-secret endpoint: {debug}"
        );
        assert!(debug.contains("ws"), "Debug dropped the prefix: {debug}");
    }

    /// RISK (mutation-discriminating): the REST-local needle list used to be an EXACT match on
    /// `credential` / `token` / `client_secret` only. Every key below is a real vended-credential
    /// key that such a list would MISS entirely — they are covered only because redaction now
    /// routes through the canonical superset `iceberg::io::is_secret_prop_key`.
    /// RED-able: swap `is_secret_prop_key` in `RedactedProps` for the old exact-match list.
    #[test]
    fn test_prop_redaction_uses_canonical_needle_superset() {
        let missed_by_exact_match_list = [
            "s3.secret-access-key",
            "s3.session-token",
            "s3.access-key-id",
            "gcs.oauth2.token",
            "adls.client-secret",
            "adls.connection-string",
            "oss.access-key-secret",
            "s3.sse.md5",
        ];

        for key in missed_by_exact_match_list {
            assert!(
                is_secret_prop_key(key),
                "canonical needle test must cover the vended-credential key `{key}`"
            );

            let props = HashMap::from([(key.to_string(), SECRET_SENTINEL.to_string())]);
            let debug = format!("{:?}", RedactedProps(&props));
            assert!(
                !debug.contains(SECRET_SENTINEL),
                "`{key}` value leaked: {debug}"
            );
            assert!(debug.contains(REDACTED), "`{key}` was not masked: {debug}");
        }

        // Anti-over-redaction: keys with no secret needle must render their real value.
        for key in ["s3.endpoint", "region", "prefix", "warehouse"] {
            let props = HashMap::from([(key.to_string(), "plain-value".to_string())]);
            let debug = format!("{:?}", RedactedProps(&props));
            assert!(
                debug.contains("plain-value"),
                "non-secret key `{key}` was over-redacted: {debug}"
            );
            assert!(
                !debug.contains(REDACTED),
                "non-secret key `{key}` was masked: {debug}"
            );
        }
    }

    /// RISK: `StorageCredential.config` IS the vended credential.
    /// RED-able: restore `#[derive(Debug)]` on `StorageCredential`.
    #[test]
    fn test_storage_credential_debug_redacts_config() {
        let credential = StorageCredential {
            prefix: "s3://bucket/warehouse/default.db/t".to_string(),
            config: HashMap::from([
                ("s3.access-key-id".to_string(), SECRET_SENTINEL.to_string()),
                (
                    "s3.secret-access-key".to_string(),
                    SECRET_SENTINEL.to_string(),
                ),
                ("s3.session-token".to_string(), SECRET_SENTINEL.to_string()),
                ("s3.region".to_string(), "us-east-1".to_string()),
            ]),
        };

        let debug = format!("{credential:?}");

        assert!(
            !debug.contains(SECRET_SENTINEL),
            "Debug leaked a vended credential: {debug}"
        );
        assert!(
            debug.contains(REDACTED),
            "expected redaction marker: {debug}"
        );
        // Anti-over-redaction: the prefix drives credential selection and must stay readable.
        assert!(
            debug.contains("s3://bucket/warehouse/default.db/t"),
            "Debug dropped the credential prefix: {debug}"
        );
        assert!(
            debug.contains("us-east-1"),
            "Debug dropped the non-secret region: {debug}"
        );
    }

    /// RISK: `LoadTableResult` carries BOTH credential channels the REST spec defines — the
    /// `config` overlay and `storage-credentials`. Both must be masked in one `{:?}`.
    /// RED-able: restore `#[derive(Debug)]` on `LoadTableResult` (or on `StorageCredential`,
    /// which this renders through).
    #[test]
    fn test_load_table_result_debug_redacts_config_and_storage_credentials() {
        let result = LoadTableResult {
            metadata_location: Some(
                "s3://bucket/warehouse/default.db/t/metadata/00001-abc.metadata.json".to_string(),
            ),
            metadata: minimal_table_metadata(),
            config: HashMap::from([
                (
                    "s3.secret-access-key".to_string(),
                    SECRET_SENTINEL.to_string(),
                ),
                (
                    "s3.endpoint".to_string(),
                    "https://s3.example.test".to_string(),
                ),
            ]),
            storage_credentials: Some(vec![StorageCredential {
                prefix: "s3://bucket".to_string(),
                config: HashMap::from([(
                    "s3.session-token".to_string(),
                    SECRET_SENTINEL.to_string(),
                )]),
            }]),
        };

        let debug = format!("{result:?}");

        assert!(
            !debug.contains(SECRET_SENTINEL),
            "Debug leaked a table credential: {debug}"
        );
        assert!(
            debug.contains(REDACTED),
            "expected redaction marker: {debug}"
        );
        // Anti-over-redaction: location and non-secret config still render.
        assert!(
            debug.contains("00001-abc.metadata.json"),
            "Debug dropped the metadata location: {debug}"
        );
        assert!(
            debug.contains("https://s3.example.test"),
            "Debug dropped the non-secret endpoint: {debug}"
        );
    }

    /// RISK: `LoadViewResult.config` is the same server-vended overlay channel as the table one.
    /// RED-able: restore `#[derive(Debug)]` on `LoadViewResult`.
    #[test]
    fn test_load_view_result_debug_redacts_config() {
        let json = serde_json::json!({
            "metadata-location": "s3://bucket/warehouse/default.db/v/metadata/00001-abc.metadata.json",
            "metadata": {
                "view-uuid": "fa6506c3-7681-40c8-86dc-e36561f83385",
                "format-version": 1,
                "location": "s3://bucket/warehouse/default.db/v",
                "current-version-id": 1,
                "properties": {},
                "versions": [ {
                    "version-id": 1,
                    "timestamp-ms": 1573518431292i64,
                    "schema-id": 1,
                    "default-namespace": [ "default" ],
                    "summary": {},
                    "representations": [ {
                        "type": "sql", "sql": "SELECT 1 AS c", "dialect": "spark"
                    } ]
                } ],
                "schemas": [ {
                    "schema-id": 1, "type": "struct",
                    "fields": [ { "id": 1, "name": "c", "required": false, "type": "int" } ]
                } ],
                "version-log": [ { "timestamp-ms": 1573518431292i64, "version-id": 1 } ]
            },
            "config": {
                "gcs.oauth2.token": SECRET_SENTINEL,
                "gcs.project-id": "visible-project"
            }
        });

        let result: LoadViewResult =
            serde_json::from_value(json).expect("LoadViewResult must deserialize");

        let debug = format!("{result:?}");

        assert!(
            !debug.contains(SECRET_SENTINEL),
            "Debug leaked a view credential: {debug}"
        );
        assert!(
            debug.contains(REDACTED),
            "expected redaction marker: {debug}"
        );
        assert!(
            debug.contains("visible-project"),
            "Debug dropped the non-secret config entry: {debug}"
        );
    }

    /// RISK: `NamespaceResponse.properties` is a server-returned property map.
    /// RED-able: restore `#[derive(Debug)]` on `NamespaceResponse`.
    #[test]
    fn test_namespace_response_debug_redacts_secret_properties() {
        let response = NamespaceResponse {
            namespace: NamespaceIdent::new("ns".to_string()),
            properties: HashMap::from([
                ("credential".to_string(), SECRET_SENTINEL.to_string()),
                ("owner".to_string(), "data-eng".to_string()),
            ]),
        };

        let debug = format!("{response:?}");

        assert!(
            !debug.contains(SECRET_SENTINEL),
            "Debug leaked a namespace credential: {debug}"
        );
        assert!(
            debug.contains(REDACTED),
            "expected redaction marker: {debug}"
        );
        assert!(
            debug.contains("data-eng"),
            "Debug dropped the non-secret owner property: {debug}"
        );
    }

    /// RISK: the namespace WRITE paths carry the same values in the other direction.
    /// RED-able: restore `#[derive(Debug)]` on either request type.
    #[test]
    fn test_namespace_request_debug_redacts_secret_properties() {
        let create = CreateNamespaceRequest {
            namespace: NamespaceIdent::new("ns".to_string()),
            properties: HashMap::from([
                ("credential".to_string(), SECRET_SENTINEL.to_string()),
                ("owner".to_string(), "data-eng".to_string()),
            ]),
        };
        let create_debug = format!("{create:?}");
        assert!(
            !create_debug.contains(SECRET_SENTINEL),
            "CreateNamespaceRequest Debug leaked a credential: {create_debug}"
        );
        assert!(
            create_debug.contains("data-eng"),
            "CreateNamespaceRequest Debug dropped the non-secret property: {create_debug}"
        );

        let update = UpdateNamespacePropertiesRequest {
            removals: Some(vec!["stale-owner".to_string()]),
            updates: HashMap::from([
                ("s3.session-token".to_string(), SECRET_SENTINEL.to_string()),
                ("owner".to_string(), "data-eng".to_string()),
            ]),
        };
        let update_debug = format!("{update:?}");
        assert!(
            !update_debug.contains(SECRET_SENTINEL),
            "UpdateNamespacePropertiesRequest Debug leaked a credential: {update_debug}"
        );
        assert!(
            update_debug.contains(REDACTED),
            "expected redaction marker: {update_debug}"
        );
        // Anti-over-redaction: removals are key-only and must stay fully visible.
        assert!(
            update_debug.contains("stale-owner"),
            "UpdateNamespacePropertiesRequest Debug dropped removals: {update_debug}"
        );
        assert!(
            update_debug.contains("data-eng"),
            "UpdateNamespacePropertiesRequest Debug dropped the non-secret property: {update_debug}"
        );
    }

    /// RISK: table/view CREATE requests carry operator-authored properties, which are a documented
    /// place FileIO credentials get set.
    /// RED-able: restore `#[derive(Debug)]` on either request type.
    #[test]
    fn test_create_table_and_view_request_debug_redact_properties() {
        let table = CreateTableRequest {
            name: "t".to_string(),
            location: Some("s3://bucket/warehouse/default.db/t".to_string()),
            schema: minimal_schema(),
            partition_spec: None,
            write_order: None,
            stage_create: Some(false),
            properties: HashMap::from([
                (
                    "s3.secret-access-key".to_string(),
                    SECRET_SENTINEL.to_string(),
                ),
                ("write.format.default".to_string(), "parquet".to_string()),
            ]),
        };
        let table_debug = format!("{table:?}");
        assert!(
            !table_debug.contains(SECRET_SENTINEL),
            "CreateTableRequest Debug leaked a credential: {table_debug}"
        );
        assert!(
            table_debug.contains(REDACTED),
            "expected redaction marker: {table_debug}"
        );
        // Anti-over-redaction: name/location/non-secret properties still render.
        assert!(
            table_debug.contains("s3://bucket/warehouse/default.db/t"),
            "CreateTableRequest Debug dropped the location: {table_debug}"
        );
        assert!(
            table_debug.contains("parquet"),
            "CreateTableRequest Debug dropped the non-secret property: {table_debug}"
        );

        let view_json = serde_json::json!({
            "name": "v",
            "location": "s3://bucket/warehouse/default.db/v",
            "view-version": {
                "version-id": 1,
                "timestamp-ms": 1573518431292i64,
                "schema-id": 0,
                "default-namespace": [ "default" ],
                "summary": {},
                "representations": [ {
                    "type": "sql", "sql": "SELECT 1 AS x", "dialect": "spark"
                } ]
            },
            "schema": {
                "schema-id": 0, "type": "struct",
                "fields": [ { "id": 1, "name": "x", "required": true, "type": "long" } ]
            },
            "properties": {
                "adls.connection-string": SECRET_SENTINEL,
                "comment": "visible-comment"
            }
        });
        let view: CreateViewRequest =
            serde_json::from_value(view_json).expect("CreateViewRequest must deserialize");
        let view_debug = format!("{view:?}");
        assert!(
            !view_debug.contains(SECRET_SENTINEL),
            "CreateViewRequest Debug leaked a credential: {view_debug}"
        );
        assert!(
            view_debug.contains(REDACTED),
            "expected redaction marker: {view_debug}"
        );
        assert!(
            view_debug.contains("visible-comment"),
            "CreateViewRequest Debug dropped the non-secret property: {view_debug}"
        );
    }

    #[test]
    fn test_namespace_response_serde() {
        let json = serde_json::json!({
            "namespace": ["nested", "ns"],
            "properties": {
                "key1": "value1",
                "key2": "value2"
            }
        });
        let ns_response: NamespaceResponse =
            serde_json::from_value(json.clone()).expect("Deserialization failed");
        assert_eq!(ns_response, NamespaceResponse {
            namespace: NamespaceIdent::from_vec(vec!["nested".to_string(), "ns".to_string()])
                .unwrap(),
            properties: HashMap::from([
                ("key1".to_string(), "value1".to_string()),
                ("key2".to_string(), "value2".to_string()),
            ]),
        });
        assert_eq!(
            serde_json::to_value(&ns_response).expect("Serialization failed"),
            json
        );

        // Without properties
        let json_no_props = serde_json::json!({
            "namespace": ["db", "schema"]
        });
        let ns_response_no_props: NamespaceResponse =
            serde_json::from_value(json_no_props.clone()).expect("Deserialization failed");
        assert_eq!(ns_response_no_props, NamespaceResponse {
            namespace: NamespaceIdent::from_vec(vec!["db".to_string(), "schema".to_string()])
                .unwrap(),
            properties: HashMap::new(),
        });
        assert_eq!(
            serde_json::to_value(&ns_response_no_props).expect("Serialization failed"),
            json_no_props
        );
    }

    // RISK: the LoadViewResult wire shape must match Java's `LoadViewResponse`
    // (`metadata-location` / `metadata` / `config`). A wrong key (e.g. snake_case `metadata_location`)
    // would silently fail to deserialize a real server's load-view / create-view response, and the
    // embedded `metadata` must parse as a full `ViewMetadata` (versions, schemas, version-log).
    #[test]
    fn test_load_view_result_serde() {
        let json = serde_json::json!({
            "metadata-location": "s3://bucket/warehouse/default.db/event_agg/metadata/00001-abc.metadata.json",
            "metadata": {
                "view-uuid": "fa6506c3-7681-40c8-86dc-e36561f83385",
                "format-version": 1,
                "location": "s3://bucket/warehouse/default.db/event_agg",
                "current-version-id": 1,
                "properties": { "comment": "Daily event counts" },
                "versions": [ {
                    "version-id": 1,
                    "timestamp-ms": 1573518431292i64,
                    "schema-id": 1,
                    "default-namespace": [ "default" ],
                    "summary": { "engine-name": "Spark" },
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
                    "timestamp-ms": 1573518431292i64,
                    "version-id": 1
                } ]
            },
            "config": { "key": "value" }
        });

        let result: LoadViewResult =
            serde_json::from_value(json).expect("LoadViewResult deserialization failed");
        assert_eq!(
            result.metadata_location,
            "s3://bucket/warehouse/default.db/event_agg/metadata/00001-abc.metadata.json"
        );
        assert_eq!(
            result.metadata.uuid().to_string(),
            "fa6506c3-7681-40c8-86dc-e36561f83385"
        );
        assert_eq!(result.metadata.current_version_id(), 1);
        assert_eq!(result.metadata.versions().count(), 1);
        assert_eq!(result.config.get("key"), Some(&"value".to_string()));

        // Round-trips back to a value with the same kebab-case keys.
        let reserialized =
            serde_json::to_value(&result).expect("LoadViewResult serialization failed");
        assert!(reserialized.get("metadata-location").is_some());
        assert!(reserialized.get("metadata").is_some());
        assert!(reserialized.get("config").is_some());
    }

    // RISK: the CreateViewRequest wire shape must match Java's `CreateViewRequest`
    // (`name` / `location` / `view-version` / `schema` / `properties`). The `view-version` field
    // (kebab-case) carries a full `ViewVersion`; a snake_case `view_version` or a missing field
    // would be rejected by a real server.
    #[test]
    fn test_create_view_request_serde() {
        let json = serde_json::json!({
            "name": "event_agg",
            "location": "s3://bucket/warehouse/default.db/event_agg",
            "view-version": {
                "version-id": 1,
                "timestamp-ms": 1573518431292i64,
                "schema-id": 1,
                "default-namespace": [ "default" ],
                "summary": {},
                "representations": [ {
                    "type": "sql",
                    "sql": "SELECT 1 AS event_count",
                    "dialect": "spark"
                } ]
            },
            "schema": {
                "schema-id": 1,
                "type": "struct",
                "fields": [ {
                    "id": 1,
                    "name": "event_count",
                    "required": false,
                    "type": "int"
                } ]
            },
            "properties": { "comment": "daily" }
        });

        let request: CreateViewRequest =
            serde_json::from_value(json.clone()).expect("CreateViewRequest deserialization failed");
        assert_eq!(request.name, "event_agg");
        assert_eq!(
            request.location.as_deref(),
            Some("s3://bucket/warehouse/default.db/event_agg")
        );
        assert_eq!(request.view_version.version_id(), 1);
        assert_eq!(request.schema.schema_id(), 1);
        assert_eq!(
            request.properties.get("comment"),
            Some(&"daily".to_string())
        );

        // Round-trip is value-stable (kebab-case `view-version` survives).
        let reserialized =
            serde_json::to_value(&request).expect("CreateViewRequest serialization failed");
        assert_eq!(reserialized, json);
    }

    // RISK: the CommitViewRequest reuses Java's `UpdateTableRequest` shape
    // (`identifier` / `requirements` / `updates`) but carries VIEW requirements/updates. The view
    // requirement tag (`assert-view-uuid`) and the view update action tags must serialize per the
    // REST spec; a table-requirement leak would make a real server reject the commit.
    #[test]
    fn test_commit_view_request_serde() {
        let json = serde_json::json!({
            "identifier": { "namespace": ["default"], "name": "event_agg" },
            "requirements": [ {
                "type": "assert-view-uuid",
                "uuid": "fa6506c3-7681-40c8-86dc-e36561f83385"
            } ],
            "updates": [ {
                "action": "set-properties",
                "updates": { "comment": "daily counts" }
            } ]
        });

        let request: CommitViewRequest =
            serde_json::from_value(json.clone()).expect("CommitViewRequest deserialization failed");
        assert_eq!(request.requirements.len(), 1);
        assert!(matches!(
            request.requirements[0],
            ViewRequirement::UuidMatch { .. }
        ));
        assert_eq!(request.updates.len(), 1);
        assert!(matches!(
            request.updates[0],
            ViewUpdate::SetProperties { .. }
        ));

        // Round-trip is value-stable (view tags `assert-view-uuid` / `set-properties` survive).
        let reserialized =
            serde_json::to_value(&request).expect("CommitViewRequest serialization failed");
        assert_eq!(reserialized, json);
    }
}
