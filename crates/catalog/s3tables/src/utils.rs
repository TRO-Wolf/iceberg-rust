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

use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_sdk_s3tables::config::Credentials;
use iceberg::spec::{TableMetadata, TableProperties};
use iceberg::{Error, ErrorKind, Result};

/// Property aws profile name
pub const AWS_PROFILE_NAME: &str = "profile_name";
/// Property aws region
pub const AWS_REGION_NAME: &str = "region_name";
/// Property aws access key
pub const AWS_ACCESS_KEY_ID: &str = "aws_access_key_id";
/// Property aws secret access key
pub const AWS_SECRET_ACCESS_KEY: &str = "aws_secret_access_key";
/// Property aws session token
pub const AWS_SESSION_TOKEN: &str = "aws_session_token";

/// Creates an aws sdk configuration based on
/// provided properties and an optional endpoint URL.
pub(crate) async fn create_sdk_config(
    properties: &HashMap<String, String>,
    endpoint_url: Option<String>,
) -> SdkConfig {
    let mut config = aws_config::defaults(BehaviorVersion::latest());

    if properties.is_empty() {
        return config.load().await;
    }

    if let Some(endpoint_url) = endpoint_url {
        config = config.endpoint_url(endpoint_url);
    }

    if let (Some(access_key), Some(secret_key)) = (
        properties.get(AWS_ACCESS_KEY_ID),
        properties.get(AWS_SECRET_ACCESS_KEY),
    ) {
        let session_token = properties.get(AWS_SESSION_TOKEN).cloned();
        let credentials_provider =
            Credentials::new(access_key, secret_key, session_token, None, "properties");

        config = config.credentials_provider(credentials_provider)
    };

    if let Some(profile_name) = properties.get(AWS_PROFILE_NAME) {
        config = config.profile_name(profile_name);
    }

    if let Some(region_name) = properties.get(AWS_REGION_NAME) {
        let region = Region::new(region_name.clone());
        config = config.region(region);
    }

    config.load().await
}

fn under_warehouse(path: &str, warehouse: &str) -> bool {
    let warehouse = warehouse.trim_end_matches('/');
    let path = path.trim_end_matches('/');
    path == warehouse || path.starts_with(&format!("{warehouse}/"))
}

pub(crate) fn ensure_write_paths_under_warehouse(
    metadata: &TableMetadata,
    metadata_location: &str,
) -> Result<()> {
    let warehouse = metadata.location();
    if !under_warehouse(metadata_location, warehouse) {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "S3 Tables owns the table warehouse: metadata location {metadata_location} is outside {warehouse}"
            ),
        ));
    }
    for key in [
        TableProperties::PROPERTY_WRITE_METADATA_LOCATION,
        TableProperties::PROPERTY_WRITE_DATA_LOCATION,
    ] {
        if let Some(value) = metadata.properties().get(key)
            && !under_warehouse(value, warehouse)
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("S3 Tables owns the table warehouse: {key}={value} is outside {warehouse}"),
            ));
        }
    }
    Ok(())
}

pub(crate) fn ensure_no_write_path_override(properties: &HashMap<String, String>) -> Result<()> {
    for key in [
        TableProperties::PROPERTY_WRITE_METADATA_LOCATION,
        TableProperties::PROPERTY_WRITE_DATA_LOCATION,
    ] {
        if let Some(value) = properties.get(key) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "S3 Tables generates the table warehouse: {key}={value} cannot be under it and cannot be honored"
                ),
            ));
        }
    }
    Ok(())
}
