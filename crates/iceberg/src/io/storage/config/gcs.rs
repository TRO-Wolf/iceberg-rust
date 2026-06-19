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

//! Google Cloud Storage configuration.
//!
//! This module provides configuration constants and types for Google Cloud Storage.
//! Reference: https://github.com/apache/iceberg/blob/main/gcp/src/main/java/org/apache/iceberg/gcp/GCPProperties.java

use std::fmt::{self, Debug, Formatter};

use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use super::StorageConfig;
use crate::Result;
use crate::io::is_truthy;

/// Renders an optional secret for `Debug`: `Some(_)` becomes the redaction marker
/// `"***"` (preserving presence) and `None` stays `None` (the value is never printed).
fn redact_secret(secret: &Option<String>) -> Option<&'static str> {
    secret.as_ref().map(|_| "***")
}

/// Google Cloud Project ID.
pub const GCS_PROJECT_ID: &str = "gcs.project-id";
/// Google Cloud Storage endpoint.
pub const GCS_SERVICE_PATH: &str = "gcs.service.path";
/// Google Cloud user project.
pub const GCS_USER_PROJECT: &str = "gcs.user-project";
/// Allow unauthenticated requests.
pub const GCS_NO_AUTH: &str = "gcs.no-auth";
/// Google Cloud Storage credentials JSON string, base64 encoded.
///
/// E.g. base64::prelude::BASE64_STANDARD.encode(serde_json::to_string(credential).as_bytes())
pub const GCS_CREDENTIALS_JSON: &str = "gcs.credentials-json";
/// Google Cloud Storage token.
pub const GCS_TOKEN: &str = "gcs.oauth2.token";
/// Option to skip signing requests (e.g. for public buckets/folders).
pub const GCS_ALLOW_ANONYMOUS: &str = "gcs.allow-anonymous";
/// Option to skip loading the credential from GCE metadata server.
pub const GCS_DISABLE_VM_METADATA: &str = "gcs.disable-vm-metadata";
/// Option to skip loading configuration from config file and the env.
pub const GCS_DISABLE_CONFIG_LOAD: &str = "gcs.disable-config-load";

/// Google Cloud Storage configuration.
///
/// This struct contains all the configuration options for connecting to Google Cloud Storage.
/// Use the builder pattern via `GcsConfig::builder()` to construct instances.
/// ```
#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize, TypedBuilder)]
pub struct GcsConfig {
    /// Google Cloud Project ID.
    #[builder(default, setter(strip_option, into))]
    pub project_id: Option<String>,
    /// GCS service endpoint.
    #[builder(default, setter(strip_option, into))]
    pub endpoint: Option<String>,
    /// User project for requester pays buckets.
    #[builder(default, setter(strip_option, into))]
    pub user_project: Option<String>,
    /// Credentials JSON (base64 encoded).
    #[builder(default, setter(strip_option, into))]
    pub credential: Option<String>,
    /// OAuth2 token.
    #[builder(default, setter(strip_option, into))]
    pub token: Option<String>,
    /// Allow anonymous access.
    #[builder(default)]
    pub allow_anonymous: bool,
    /// Disable VM metadata.
    #[builder(default)]
    pub disable_vm_metadata: bool,
    /// Disable config load.
    #[builder(default)]
    pub disable_config_load: bool,
}

impl Debug for GcsConfig {
    /// Hand-written so the secret fields (`credential`, `token`) are redacted:
    /// their presence is preserved as `"***"` but the value is never printed.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcsConfig")
            .field("project_id", &self.project_id)
            .field("endpoint", &self.endpoint)
            .field("user_project", &self.user_project)
            .field("credential", &redact_secret(&self.credential))
            .field("token", &redact_secret(&self.token))
            .field("allow_anonymous", &self.allow_anonymous)
            .field("disable_vm_metadata", &self.disable_vm_metadata)
            .field("disable_config_load", &self.disable_config_load)
            .finish()
    }
}

impl TryFrom<&StorageConfig> for GcsConfig {
    type Error = crate::Error;

    fn try_from(config: &StorageConfig) -> Result<Self> {
        let props = config.props();

        let mut cfg = GcsConfig::default();

        if let Some(project_id) = props.get(GCS_PROJECT_ID) {
            cfg.project_id = Some(project_id.clone());
        }
        if let Some(endpoint) = props.get(GCS_SERVICE_PATH) {
            cfg.endpoint = Some(endpoint.clone());
        }
        if let Some(user_project) = props.get(GCS_USER_PROJECT) {
            cfg.user_project = Some(user_project.clone());
        }
        if let Some(credential) = props.get(GCS_CREDENTIALS_JSON) {
            cfg.credential = Some(credential.clone());
        }
        if let Some(token) = props.get(GCS_TOKEN) {
            cfg.token = Some(token.clone());
        }

        // GCS_NO_AUTH enables all anonymous/no-auth options
        if props.get(GCS_NO_AUTH).is_some() {
            cfg.allow_anonymous = true;
            cfg.disable_vm_metadata = true;
            cfg.disable_config_load = true;
        }

        if let Some(allow_anonymous) = props.get(GCS_ALLOW_ANONYMOUS)
            && is_truthy(allow_anonymous.to_lowercase().as_str())
        {
            cfg.allow_anonymous = true;
        }
        if let Some(disable_vm_metadata) = props.get(GCS_DISABLE_VM_METADATA)
            && is_truthy(disable_vm_metadata.to_lowercase().as_str())
        {
            cfg.disable_vm_metadata = true;
        }
        if let Some(disable_config_load) = props.get(GCS_DISABLE_CONFIG_LOAD)
            && is_truthy(disable_config_load.to_lowercase().as_str())
        {
            cfg.disable_config_load = true;
        }

        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcs_config_builder() {
        let config = GcsConfig::builder()
            .project_id("my-project")
            .credential("base64-creds")
            .endpoint("http://localhost:4443")
            .build();

        assert_eq!(config.project_id.as_deref(), Some("my-project"));
        assert_eq!(config.credential.as_deref(), Some("base64-creds"));
        assert_eq!(config.endpoint.as_deref(), Some("http://localhost:4443"));
    }

    #[test]
    fn test_gcs_config_from_storage_config() {
        let storage_config = StorageConfig::new()
            .with_prop(GCS_PROJECT_ID, "my-project")
            .with_prop(GCS_CREDENTIALS_JSON, "base64-creds")
            .with_prop(GCS_SERVICE_PATH, "http://localhost:4443");

        let gcs_config = GcsConfig::try_from(&storage_config).unwrap();

        assert_eq!(gcs_config.project_id.as_deref(), Some("my-project"));
        assert_eq!(gcs_config.credential.as_deref(), Some("base64-creds"));
        assert_eq!(
            gcs_config.endpoint.as_deref(),
            Some("http://localhost:4443")
        );
    }

    #[test]
    fn test_gcs_config_no_auth() {
        let storage_config = StorageConfig::new().with_prop(GCS_NO_AUTH, "true");

        let gcs_config = GcsConfig::try_from(&storage_config).unwrap();

        assert!(gcs_config.allow_anonymous);
        assert!(gcs_config.disable_vm_metadata);
        assert!(gcs_config.disable_config_load);
    }

    #[test]
    fn test_gcs_config_allow_anonymous() {
        let storage_config = StorageConfig::new().with_prop(GCS_ALLOW_ANONYMOUS, "true");

        let gcs_config = GcsConfig::try_from(&storage_config).unwrap();

        assert!(gcs_config.allow_anonymous);
        assert!(!gcs_config.disable_vm_metadata);
    }

    #[test]
    fn test_gcs_config_debug_redacts_secrets() {
        let secret = "SECRET_VALUE_DO_NOT_LEAK";
        let config = GcsConfig::builder()
            .project_id("my-project")
            .credential(secret)
            .token(secret)
            .build();

        let debug = format!("{config:?}");

        assert!(
            !debug.contains(secret),
            "Debug output leaked a secret value: {debug}"
        );
        assert!(debug.contains("***"), "expected redaction marker: {debug}");
        assert!(
            debug.contains("my-project"),
            "Debug dropped non-secret fields: {debug}"
        );
    }
}
