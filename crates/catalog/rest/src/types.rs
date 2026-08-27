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

use iceberg::io::RedactedProps;
use iceberg::spec::{
    Schema, SortOrder, TableMetadata, UnboundPartitionSpec, ViewMetadata, ViewVersion,
};
use iceberg::{
    Error, ErrorKind, Namespace, NamespaceIdent, TableIdent, TableRequirement, TableUpdate,
    ViewRequirement, ViewUpdate,
};
use serde_derive::{Deserialize, Serialize};

// Secret redaction for `Debug`.
//
// Wire types here carry live credentials: the OAuth `access_token`, the vended storage
// credentials, and the property maps the catalog copies into `FileIO` props. A derived `Debug`
// prints all of it at any `{:?}` site. Every secret-bearing type below hand-writes `Debug`.
//
// Property maps redact PER KEY through `iceberg::io::RedactedProps` and `is_secret_prop_key`.
// Keys stay visible and only values mask. One authoritative needle list serves every crate, so no
// copy drifts. It is deliberately a superset: over-redaction is the safe direction here.
//
// These keep a derived `Debug`, holding no value map and no secret: `ListNamespaceResponse`,
// `ListTablesResponse`, `RenameTableRequest`, `RegisterTableRequest`,
// `UpdateNamespacePropertiesResponse`. So do `ErrorModel`, `ErrorResponse`, and `OAuthError`,
// whose server-controlled free text `Display` already surfaces verbatim. Key-based redaction
// cannot mask a secret a server splices into free text.
//
// Maps that still print in clear, each reachable from a type in this module:
//
// | Map | Home | Reached through |
// |---|---|---|
// | `ViewVersion.summary` | `spec/view_version.rs` | `CreateViewRequest.view_version` |
// | `TableUpdate`/`ViewUpdate::SetProperties` | `catalog/mod.rs` | `Commit*Request.updates` |
// | `EncryptedKey.properties`, `encrypted_key_metadata` | `spec/encrypted_key.rs` | `TableMetadata` |
// | `Snapshot.summary.additional_properties` | `spec/snapshot.rs` | `TableMetadata` |
// | `StatisticsFile.key_metadata`, `blob_metadata[*].properties` | `spec/statistic_file.rs` | `TableMetadata` |
//
// Closing any `SetProperties` row needs BOTH ends: masking the core enum alone leaves the REST
// request type printing the same map. The `TableMetadata` rows are maintained authoritatively on
// `impl Debug for TableMetadata`; this list mirrors it.

/// Marker written in place of a redacted secret value. Its presence also signals a populated
/// field. It is re-exported from the core crate so no crate drifts on the marker it emits.
const REDACTED: &str = iceberg::io::REDACTED_PROP_VALUE;

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct CatalogConfig {
    pub(super) overrides: HashMap<String, String>,
    pub(super) defaults: HashMap<String, String>,
}

impl std::fmt::Debug for CatalogConfig {
    /// Hand-written: `defaults` and `overrides` reach `FileIO` and carry vended credentials.
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
    /// Hand-written: `access_token` is raw bearer-secret material. The other fields stay
    /// readable, so a failing exchange is diagnosable. [`REDACTED`] signals presence only. The
    /// token LENGTH is deliberately not emitted: a length is a weak oracle on secret material.
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
    /// Hand-written: `properties` is a server-returned map. Entries redact per key.
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
    /// Hand-written: `properties` may carry operator-set credentials. Entries redact per key.
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
    /// Opaque pagination token. When present, pass it back to fetch the next page.
    pub next_page_token: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Update namespace properties. An absent property is left alone. A server may not support them.
pub struct UpdateNamespacePropertiesRequest {
    /// List of property keys to remove from the namespace
    pub removals: Option<Vec<String>>,
    /// Map of property keys to values to set or update on the namespace
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub updates: HashMap<String, String>,
}

impl std::fmt::Debug for UpdateNamespacePropertiesRequest {
    /// Hand-written: `updates` redacts per key. `removals` is key-only and stays readable.
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
    /// Requested removals the namespace did not hold. A server need not report them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub missing: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Response containing a list of table identifiers, with optional pagination support.
pub struct ListTablesResponse {
    /// List of table identifiers under the requested namespace
    pub identifiers: Vec<TableIdent>,
    /// Opaque pagination token. When present, pass it back to fetch the next page.
    #[serde(default)]
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Request to rename a table. A cross-namespace move is valid, but a server need not support it.
pub struct RenameTableRequest {
    /// Current table identifier to rename
    pub source: TableIdent,
    /// New table identifier to rename to
    pub destination: TableIdent,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Result returned when a table is loaded or created. A staged create transaction returns
/// uncommitted `metadata` and no `metadata_location`. `config` carries table-specific
/// configuration for the table's HTTP client and `FileIO`.
pub struct LoadTableResult {
    /// May be null if the table is staged as part of a transaction
    pub metadata_location: Option<String>,
    /// The table's full metadata
    pub metadata: TableMetadata,
    /// Table-specific configuration overriding catalog configuration
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub config: HashMap<String, String>,
    /// Storage credentials for the table data. Prefer these over the `config` credentials.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_credentials: Option<Vec<StorageCredential>>,
}

impl std::fmt::Debug for LoadTableResult {
    /// Hand-written: the spec lets a server vend credentials through `config`, which redacts
    /// per key. `storage_credentials` renders through [`StorageCredential`]'s own `Debug`.
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
/// Storage credential for one location prefix. Choose the longest matching prefix.
pub struct StorageCredential {
    /// Storage location prefix where this credential is relevant
    pub prefix: String,
    /// Configuration map containing credential information
    pub config: HashMap<String, String>,
}

impl std::fmt::Debug for StorageCredential {
    /// Hand-written: `config` IS the credential, so values redact. `prefix` stays readable.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageCredential")
            .field("prefix", &self.prefix)
            .field("config", &RedactedProps(&self.config))
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Request to create a new table. `stage_create` false creates it at once. True returns
/// initialized metadata instead, and the commit endpoint completes the create transaction.
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
    /// Hand-written: `properties` may carry FileIO credentials, so entries redact per key.
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
/// Request to commit updates to a table. Requirements are assertions validated before the
/// commit. A `stage-create` transaction commits here and must include every change, table
/// initialization included.
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
/// Response returned when a table is updated. Compare `metadata-location` to detect a change.
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

// View shapes mirror the REST OpenAPI view routes and Java's `CreateViewRequest`,
// `LoadViewResponse`, and `UpdateTableRequest`, which Java reuses for the view commit.

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Request to create a new view. Mirrors Java `CreateViewRequest`.
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
    /// Hand-written: `properties` redacts per key. `view_version.summary` still prints clear.
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
/// Result returned when a view is loaded or created. Mirrors Java `LoadViewResponse`.
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
    /// Hand-written: `config` is the same vended channel as `LoadTableResult.config`.
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
/// Request to commit updates to a view. Java reuses `UpdateTableRequest` here, so the wire shape
/// is `identifier`, `requirements`, `updates`. Both carry VIEW entries, not table ones.
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
    use iceberg::io::is_secret_prop_key;

    use super::*;

    // `Debug` must never render a secret VALUE.
    //
    // Each test below feeds a synthetic sentinel into a secret field and asserts `{:?}` omits it.
    // Each also asserts a non-secret sibling still renders: a `Debug` printing nothing would pass
    // a leak-only assertion vacuously. The mutation each discriminates is named on the test.

    /// Sentinel for secret material. Not shaped like a credential, so no scanner trips on it.
    const SECRET_SENTINEL: &str = "SENTINEL_MUST_NOT_APPEAR_IN_DEBUG";

    /// Minimal complete v2 table metadata, so `LoadTableResult` needs no fixture file.
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

    /// Risk: `access_token` is bearer-secret material. Mutation: restore the derived `Debug`.
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
        // The non-secret OAuth metadata must stay diagnosable.
        assert!(
            debug.contains("bearer"),
            "Debug dropped token_type: {debug}"
        );
        assert!(debug.contains("3600"), "Debug dropped expires_in: {debug}");
    }

    /// Risk: `defaults` and `overrides` reach `FileIO`. Mutation: restore the derived `Debug`.
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
        // Keys stay visible and non-secret values still render.
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

    /// Risk: an exact-match needle list on `credential`, `token`, and `client_secret` misses
    /// every real vended-credential key below. Only the superset `is_secret_prop_key` covers
    /// them. Mutation: swap `is_secret_prop_key` in `RedactedProps` for that exact-match list.
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

        // A key with no secret needle must render its real value.
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

    /// Risk: `config` IS the vended credential. Mutation: restore the derived `Debug`.
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
        // The prefix drives credential selection and must stay readable.
        assert!(
            debug.contains("s3://bucket/warehouse/default.db/t"),
            "Debug dropped the credential prefix: {debug}"
        );
        assert!(
            debug.contains("us-east-1"),
            "Debug dropped the non-secret region: {debug}"
        );
    }

    /// Risk: one `{:?}` must mask both credential channels the spec defines. Mutation: restore
    /// the derived `Debug` on `LoadTableResult` or on `StorageCredential`.
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
        // Location and non-secret config still render.
        assert!(
            debug.contains("00001-abc.metadata.json"),
            "Debug dropped the metadata location: {debug}"
        );
        assert!(
            debug.contains("https://s3.example.test"),
            "Debug dropped the non-secret endpoint: {debug}"
        );
    }

    /// Risk: `config` is the same vended channel as the table one. Mutation: derive `Debug`.
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

    /// Risk: `properties` is a server-returned map. Mutation: restore its derived `Debug`.
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

    /// Risk: the write paths carry the same values. Mutation: restore either derived `Debug`.
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
        // Removals are key-only and must stay fully visible.
        assert!(
            update_debug.contains("stale-owner"),
            "UpdateNamespacePropertiesRequest Debug dropped removals: {update_debug}"
        );
        assert!(
            update_debug.contains("data-eng"),
            "UpdateNamespacePropertiesRequest Debug dropped the non-secret property: {update_debug}"
        );
    }

    /// Risk: create requests carry operator-set properties. Mutation: derive either `Debug`.
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
        // Name, location, and non-secret properties still render.
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

    // Risk: the wire shape must match Java `LoadViewResponse`. A snake_case key silently fails
    // on a real server's response, and `metadata` must parse as a full `ViewMetadata`.
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

        // The round trip keeps the kebab-case keys.
        let reserialized =
            serde_json::to_value(&result).expect("LoadViewResult serialization failed");
        assert!(reserialized.get("metadata-location").is_some());
        assert!(reserialized.get("metadata").is_some());
        assert!(reserialized.get("config").is_some());
    }

    // Risk: the wire shape must match Java `CreateViewRequest`. The kebab-case `view-version`
    // carries a full `ViewVersion`. A snake_case or missing field makes a real server reject.
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

        // The round trip is value-stable and `view-version` survives.
        let reserialized =
            serde_json::to_value(&request).expect("CreateViewRequest serialization failed");
        assert_eq!(reserialized, json);
    }

    // Risk: the request reuses Java's `UpdateTableRequest` shape but carries VIEW entries.
    // A table-requirement leak makes a real server reject the commit.
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

        // The round trip is value-stable and the view tags survive.
        let reserialized =
            serde_json::to_value(&request).expect("CommitViewRequest serialization failed");
        assert_eq!(reserialized, json);
    }
}
