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

// TODO Add specific configs
//! Storage configuration for storage backends.
//!
//! This module provides configuration types for various storage backends.
//! The configuration types are designed to be used with the `StorageFactory`
//! trait to create storage instances.
//!
//! # Available Configurations
//!
//! - [`StorageConfig`]: Base configuration containing properties for storage backends
//! - [`S3Config`]: Amazon S3 specific configuration
//! - [`GcsConfig`]: Google Cloud Storage specific configuration
//! - [`OssConfig`]: Alibaba Cloud OSS specific configuration
//! - [`AzdlsConfig`]: Azure Data Lake Storage specific configuration

mod azdls;
mod gcs;
mod oss;
mod s3;

use std::collections::HashMap;

pub use azdls::*;
pub use gcs::*;
pub use oss::*;
pub use s3::*;
use serde::{Deserialize, Serialize};

/// Configuration properties for storage backends.
///
/// This struct contains only configuration properties without specifying
/// which storage backend to use. The storage type is determined by the
/// explicit factory selection.
/// ```
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct StorageConfig {
    /// Configuration properties for the storage backend
    props: HashMap<String, String>,
}

/// Substrings (matched case-insensitively) that mark a property KEY as secret-bearing, so its
/// value is redacted from [`StorageConfig`]'s `Debug`.
///
/// `StorageConfig` holds the raw, untyped property map — the REST catalog clones its full
/// runtime props (which may include `credential`/`token`/`client_secret`) into the `FileIO`
/// this config backs (`crates/catalog/rest/src/catalog.rs`, `load_file_io`), so a plain-derived
/// `Debug` — or any `Debug`-deriving struct that embeds a `FileIO`/`StorageConfig` — would print
/// live credentials.
///
/// Substring (not exact) matching keeps this a strict SUPERSET of every value the typed configs
/// (`S3Config`/`GcsConfig`/`AzdlsConfig`/`OssConfig`) and `RestCatalogConfig` hand-redact:
/// `credential` (`gcs.credentials-json`, REST `credential`), `token` (`s3.session-token`,
/// `adls.sas-token`, `gcs.oauth2.token`, REST `token`), `secret` (`s3.secret-access-key`,
/// `adls.client-secret`, `oss.access-key-secret`, REST `client_secret`), `key` (`s3.access-key-id`,
/// `s3.sse.key`, `adls.account-key`), `md5` (`s3.sse.md5` — the SSE-C customer-key digest), and
/// `connection-string` (`adls.connection-string`, which embeds the account key / SAS token).
/// `password` is carried defensively (no typed config exposes one today). Over-redaction is the
/// safe direction for a debug view.
const SECRET_PROP_KEY_SUBSTRINGS: &[&str] = &[
    "credential",
    "token",
    "secret",
    "key",
    "password",
    "md5",
    "connection-string",
];

/// Returns true if a property key holds a secret value that must be redacted from `Debug`,
/// i.e. its lowercased form contains any [`SECRET_PROP_KEY_SUBSTRINGS`] entry.
///
/// Exposed as the canonical needle test so the catalog crates' config `Debug` impls
/// (`iceberg-catalog-glue`/`-hms`/`-s3tables`/`-sql`, whose raw property maps carry the same
/// AWS-credential / DSN secrets) redact against ONE authoritative superset instead of drifting
/// copies. Keep new secret substrings in [`SECRET_PROP_KEY_SUBSTRINGS`] here so every consumer
/// inherits them.
pub fn is_secret_prop_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    SECRET_PROP_KEY_SUBSTRINGS
        .iter()
        .any(|needle| key.contains(needle))
}

impl std::fmt::Debug for StorageConfig {
    /// Hand-written so secret-bearing entries in the raw `props` map are redacted to `"***"`
    /// instead of printed in clear. Keys stay visible for diagnostics; only secret VALUES are
    /// masked. Mirrors the `RestCatalogConfig` redaction so a `{:?}`/`tracing` of a
    /// `StorageConfig` — or of any struct that embeds a `FileIO` and derives `Debug` — cannot
    /// leak credentials.
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

        f.debug_struct("StorageConfig")
            .field("props", &redacted_props)
            .finish()
    }
}

impl StorageConfig {
    /// Create a new empty StorageConfig.
    pub fn new() -> Self {
        Self {
            props: HashMap::new(),
        }
    }

    /// Create a StorageConfig from existing properties.
    ///
    /// # Arguments
    ///
    /// * `props` - Configuration properties for the storage backend
    pub fn from_props(props: HashMap<String, String>) -> Self {
        Self { props }
    }

    /// Get all configuration properties.
    pub fn props(&self) -> &HashMap<String, String> {
        &self.props
    }

    /// Get a specific configuration property by key.
    ///
    /// # Arguments
    ///
    /// * `key` - The property key to look up
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the property value if it exists.
    pub fn get(&self, key: &str) -> Option<&String> {
        self.props.get(key)
    }

    /// Add a configuration property.
    ///
    /// This is a builder-style method that returns `self` for chaining.
    ///
    /// # Arguments
    ///
    /// * `key` - The property key
    /// * `value` - The property value
    pub fn with_prop(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.props.insert(key.into(), value.into());
        self
    }

    /// Add multiple configuration properties.
    ///
    /// This is a builder-style method that returns `self` for chaining.
    ///
    /// # Arguments
    ///
    /// * `props` - An iterator of key-value pairs to add
    pub fn with_props(
        mut self,
        props: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        self.props
            .extend(props.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_config_new() {
        let config = StorageConfig::new();

        assert!(config.props().is_empty());
    }

    #[test]
    fn test_storage_config_from_props() {
        let props = HashMap::from([
            ("region".to_string(), "us-east-1".to_string()),
            ("bucket".to_string(), "my-bucket".to_string()),
        ]);
        let config = StorageConfig::from_props(props.clone());

        assert_eq!(config.props(), &props);
    }

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();

        assert!(config.props().is_empty());
    }

    #[test]
    fn test_storage_config_get() {
        let config = StorageConfig::new().with_prop("region", "us-east-1");

        assert_eq!(config.get("region"), Some(&"us-east-1".to_string()));
        assert_eq!(config.get("nonexistent"), None);
    }

    #[test]
    fn test_storage_config_with_prop() {
        let config = StorageConfig::new()
            .with_prop("region", "us-east-1")
            .with_prop("bucket", "my-bucket");

        assert_eq!(config.get("region"), Some(&"us-east-1".to_string()));
        assert_eq!(config.get("bucket"), Some(&"my-bucket".to_string()));
    }

    #[test]
    fn test_storage_config_with_props() {
        let additional_props = vec![("key1", "value1"), ("key2", "value2")];
        let config = StorageConfig::new().with_props(additional_props);

        assert_eq!(config.get("key1"), Some(&"value1".to_string()));
        assert_eq!(config.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_storage_config_clone() {
        let config = StorageConfig::new().with_prop("region", "us-east-1");
        let cloned = config.clone();

        assert_eq!(config, cloned);
        assert_eq!(cloned.get("region"), Some(&"us-east-1".to_string()));
    }

    #[test]
    fn test_storage_config_serialization_roundtrip() {
        let config = StorageConfig::new()
            .with_prop("region", "us-east-1")
            .with_prop("bucket", "my-bucket");

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: StorageConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_storage_config_clone_independence() {
        let original = StorageConfig::new().with_prop("region", "us-east-1");
        let mut cloned = original.clone();

        // Modify the clone
        cloned = cloned.with_prop("region", "eu-west-1");
        cloned = cloned.with_prop("new_key", "new_value");

        // Original should be unchanged
        assert_eq!(original.get("region"), Some(&"us-east-1".to_string()));
        assert_eq!(original.get("new_key"), None);

        // Clone should have the new values
        assert_eq!(cloned.get("region"), Some(&"eu-west-1".to_string()));
        assert_eq!(cloned.get("new_key"), Some(&"new_value".to_string()));
    }

    #[test]
    fn test_storage_config_from_props_empty() {
        let config = StorageConfig::from_props(HashMap::new());

        assert!(config.props().is_empty());
    }

    #[test]
    fn test_storage_config_serialization_empty() {
        let config = StorageConfig::new();

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: StorageConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config, deserialized);
        assert!(deserialized.props().is_empty());
    }

    /// Risk (SEC-002 / SEC-003 chain): `StorageConfig` holds the raw prop map the REST catalog
    /// clones its full runtime props into, so a plain-derived `Debug` would print live
    /// `credential`/`token`/`client_secret` (and the typed-config secrets) in clear. Pins that
    /// secret VALUES are redacted to `"***"` while their KEYS — and every non-secret value —
    /// stay visible for diagnostics. Mutation: revert to `#[derive(.., Debug, ..)]` → RED.
    #[test]
    fn test_storage_config_debug_redacts_secret_values() {
        const SECRET: &str = "SECRET_VALUE_DO_NOT_LEAK";
        let config = StorageConfig::new()
            // REST runtime props cloned into FileIO.
            .with_prop("credential", SECRET)
            .with_prop("token", SECRET)
            .with_prop("client_secret", SECRET)
            // Representative typed-config secret keys (the superset targets).
            .with_prop("s3.secret-access-key", SECRET)
            .with_prop("s3.access-key-id", SECRET)
            .with_prop("s3.session-token", SECRET)
            .with_prop("s3.sse.md5", SECRET)
            .with_prop("adls.connection-string", SECRET)
            .with_prop("gcs.credentials-json", SECRET)
            // Non-secret diagnostic props must stay visible.
            .with_prop("region", "us-east-1")
            .with_prop("s3.endpoint", "http://localhost:9000");

        let debug = format!("{config:?}");

        assert!(
            !debug.contains(SECRET),
            "StorageConfig Debug leaked a secret value: {debug}"
        );
        // Presence is still signalled: the redaction marker and the secret KEYS survive.
        assert!(debug.contains("***"), "expected redaction marker: {debug}");
        for key in [
            "credential",
            "token",
            "client_secret",
            "s3.secret-access-key",
            "s3.sse.md5",
            "adls.connection-string",
        ] {
            assert!(
                debug.contains(key),
                "expected secret key `{key}` to remain visible: {debug}"
            );
        }
        // Non-secret values are NOT redacted.
        assert!(
            debug.contains("us-east-1") && debug.contains("http://localhost:9000"),
            "Debug dropped non-secret values: {debug}"
        );
    }

    /// Guards the "strict superset" claim: every raw property key the typed configs
    /// (`S3Config`/`GcsConfig`/`AzdlsConfig`/`OssConfig`) and `RestCatalogConfig` hand-redact
    /// must be classified secret here. If a typed config adds a secret whose key escapes the
    /// substring list, this fails — forcing the list to grow with it.
    #[test]
    fn test_secret_prop_key_is_superset_of_typed_config_secrets() {
        let redacted_by_typed_configs = [
            // RestCatalogConfig SECRET_PROP_KEYS
            "credential",
            "token",
            "client_secret",
            // S3Config
            "s3.access-key-id",
            "s3.secret-access-key",
            "s3.session-token",
            "s3.sse.key",
            "s3.sse.md5",
            // GcsConfig
            "gcs.credentials-json",
            "gcs.oauth2.token",
            // AzdlsConfig
            "adls.connection-string",
            "adls.account-key",
            "adls.sas-token",
            "adls.client-secret",
            // OssConfig
            "oss.access-key-secret",
        ];
        for key in redacted_by_typed_configs {
            assert!(
                is_secret_prop_key(key),
                "typed-config secret key `{key}` must be classified secret by StorageConfig"
            );
        }
        // Case-insensitive, and a plainly non-secret key stays visible.
        assert!(is_secret_prop_key("S3.SECRET-ACCESS-KEY"));
        assert!(!is_secret_prop_key("region"));
        assert!(!is_secret_prop_key("s3.endpoint"));
    }
}
