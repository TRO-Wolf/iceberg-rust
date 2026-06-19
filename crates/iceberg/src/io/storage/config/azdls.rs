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

//! Azure Data Lake Storage configuration.
//!
//! This module provides configuration constants and types for Azure Data Lake Storage.

use std::fmt::{self, Debug, Formatter};

use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use super::StorageConfig;
use crate::Result;

/// Renders an optional secret for `Debug`: `Some(_)` becomes the redaction marker
/// `"***"` (preserving presence) and `None` stays `None` (the value is never printed).
fn redact_secret(secret: &Option<String>) -> Option<&'static str> {
    secret.as_ref().map(|_| "***")
}

/// A connection string.
///
/// Note, this string is parsed first, and any other passed adls.* properties
/// will override values from the connection string.
pub const ADLS_CONNECTION_STRING: &str = "adls.connection-string";
/// The account that you want to connect to.
pub const ADLS_ACCOUNT_NAME: &str = "adls.account-name";
/// The key to authentication against the account.
pub const ADLS_ACCOUNT_KEY: &str = "adls.account-key";
/// The shared access signature.
pub const ADLS_SAS_TOKEN: &str = "adls.sas-token";
/// The tenant-id.
pub const ADLS_TENANT_ID: &str = "adls.tenant-id";
/// The client-id.
pub const ADLS_CLIENT_ID: &str = "adls.client-id";
/// The client-secret.
pub const ADLS_CLIENT_SECRET: &str = "adls.client-secret";
/// The authority host of the service principal.
/// - required for client_credentials authentication
/// - default value: `https://login.microsoftonline.com`
pub const ADLS_AUTHORITY_HOST: &str = "adls.authority-host";

/// Azure Data Lake Storage configuration.
///
/// This struct contains all the configuration options for connecting to Azure Data Lake Storage.
/// Use the builder pattern via `AzdlsConfig::builder()` to construct instances.
/// ```
#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize, TypedBuilder)]
pub struct AzdlsConfig {
    /// Connection string.
    #[builder(default, setter(strip_option, into))]
    pub connection_string: Option<String>,
    /// Account name.
    #[builder(default, setter(strip_option, into))]
    pub account_name: Option<String>,
    /// Account key.
    #[builder(default, setter(strip_option, into))]
    pub account_key: Option<String>,
    /// SAS token.
    #[builder(default, setter(strip_option, into))]
    pub sas_token: Option<String>,
    /// Tenant ID.
    #[builder(default, setter(strip_option, into))]
    pub tenant_id: Option<String>,
    /// Client ID.
    #[builder(default, setter(strip_option, into))]
    pub client_id: Option<String>,
    /// Client secret.
    #[builder(default, setter(strip_option, into))]
    pub client_secret: Option<String>,
    /// Authority host.
    #[builder(default, setter(strip_option, into))]
    pub authority_host: Option<String>,
    /// Endpoint URL.
    #[builder(default, setter(strip_option, into))]
    pub endpoint: Option<String>,
    /// Filesystem name.
    #[builder(default, setter(into))]
    pub filesystem: String,
}

impl Debug for AzdlsConfig {
    /// Hand-written so the secret fields are redacted: their presence is preserved as
    /// `"***"` but the value is never printed. The connection string is treated as a
    /// secret because it embeds the account key and/or SAS token.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzdlsConfig")
            .field("connection_string", &redact_secret(&self.connection_string))
            .field("account_name", &self.account_name)
            .field("account_key", &redact_secret(&self.account_key))
            .field("sas_token", &redact_secret(&self.sas_token))
            .field("tenant_id", &self.tenant_id)
            .field("client_id", &self.client_id)
            .field("client_secret", &redact_secret(&self.client_secret))
            .field("authority_host", &self.authority_host)
            .field("endpoint", &self.endpoint)
            .field("filesystem", &self.filesystem)
            .finish()
    }
}

impl TryFrom<&StorageConfig> for AzdlsConfig {
    type Error = crate::Error;

    fn try_from(config: &StorageConfig) -> Result<Self> {
        let props = config.props();

        let mut cfg = AzdlsConfig::default();

        if let Some(connection_string) = props.get(ADLS_CONNECTION_STRING) {
            cfg.connection_string = Some(connection_string.clone());
        }
        if let Some(account_name) = props.get(ADLS_ACCOUNT_NAME) {
            cfg.account_name = Some(account_name.clone());
        }
        if let Some(account_key) = props.get(ADLS_ACCOUNT_KEY) {
            cfg.account_key = Some(account_key.clone());
        }
        if let Some(sas_token) = props.get(ADLS_SAS_TOKEN) {
            cfg.sas_token = Some(sas_token.clone());
        }
        if let Some(tenant_id) = props.get(ADLS_TENANT_ID) {
            cfg.tenant_id = Some(tenant_id.clone());
        }
        if let Some(client_id) = props.get(ADLS_CLIENT_ID) {
            cfg.client_id = Some(client_id.clone());
        }
        if let Some(client_secret) = props.get(ADLS_CLIENT_SECRET) {
            cfg.client_secret = Some(client_secret.clone());
        }
        if let Some(authority_host) = props.get(ADLS_AUTHORITY_HOST) {
            cfg.authority_host = Some(authority_host.clone());
        }

        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_azdls_config_builder() {
        let config = AzdlsConfig::builder()
            .account_name("myaccount")
            .account_key("my-account-key")
            .build();

        assert_eq!(config.account_name.as_deref(), Some("myaccount"));
        assert_eq!(config.account_key.as_deref(), Some("my-account-key"));
    }

    #[test]
    fn test_azdls_config_from_storage_config() {
        let storage_config = StorageConfig::new()
            .with_prop(ADLS_ACCOUNT_NAME, "myaccount")
            .with_prop(ADLS_ACCOUNT_KEY, "my-account-key");

        let azdls_config = AzdlsConfig::try_from(&storage_config).unwrap();

        assert_eq!(azdls_config.account_name.as_deref(), Some("myaccount"));
        assert_eq!(azdls_config.account_key.as_deref(), Some("my-account-key"));
    }

    #[test]
    fn test_azdls_config_with_sas_token() {
        let storage_config = StorageConfig::new()
            .with_prop(ADLS_ACCOUNT_NAME, "myaccount")
            .with_prop(ADLS_SAS_TOKEN, "my-sas-token");

        let azdls_config = AzdlsConfig::try_from(&storage_config).unwrap();

        assert_eq!(azdls_config.account_name.as_deref(), Some("myaccount"));
        assert_eq!(azdls_config.sas_token.as_deref(), Some("my-sas-token"));
    }

    #[test]
    fn test_azdls_config_with_client_credentials() {
        let storage_config = StorageConfig::new()
            .with_prop(ADLS_ACCOUNT_NAME, "myaccount")
            .with_prop(ADLS_TENANT_ID, "my-tenant")
            .with_prop(ADLS_CLIENT_ID, "my-client")
            .with_prop(ADLS_CLIENT_SECRET, "my-secret");

        let azdls_config = AzdlsConfig::try_from(&storage_config).unwrap();

        assert_eq!(azdls_config.account_name.as_deref(), Some("myaccount"));
        assert_eq!(azdls_config.tenant_id.as_deref(), Some("my-tenant"));
        assert_eq!(azdls_config.client_id.as_deref(), Some("my-client"));
        assert_eq!(azdls_config.client_secret.as_deref(), Some("my-secret"));
    }

    #[test]
    fn test_azdls_config_debug_redacts_secrets() {
        let secret = "SECRET_VALUE_DO_NOT_LEAK";
        let config = AzdlsConfig::builder()
            .account_name("myaccount")
            .account_key(secret)
            .sas_token(secret)
            .client_secret(secret)
            .connection_string(secret)
            .build();

        let debug = format!("{config:?}");

        assert!(
            !debug.contains(secret),
            "Debug output leaked a secret value: {debug}"
        );
        assert!(debug.contains("***"), "expected redaction marker: {debug}");
        // account_name is NOT secret and must stay visible for diagnostics.
        assert!(
            debug.contains("myaccount"),
            "Debug dropped non-secret fields: {debug}"
        );
    }
}
