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
//! Google Cloud Storage properties

use std::collections::HashMap;

use iceberg::io::{
    GCS_ALLOW_ANONYMOUS, GCS_CREDENTIALS_JSON, GCS_DISABLE_CONFIG_LOAD, GCS_DISABLE_VM_METADATA,
    GCS_NO_AUTH, GCS_SERVICE_PATH, GCS_TOKEN,
};
use iceberg::{Error, ErrorKind, Result};
use opendal::Operator;
use opendal::services::GcsConfig;
use url::Url;

use crate::utils::{from_opendal_error, is_truthy};

/// Parse iceberg properties to [`GcsConfig`].
pub(crate) fn gcs_config_parse(mut m: HashMap<String, String>) -> Result<GcsConfig> {
    let mut cfg = GcsConfig::default();

    if let Some(cred) = m.remove(GCS_CREDENTIALS_JSON) {
        cfg.credential = Some(cred);
    }

    if let Some(token) = m.remove(GCS_TOKEN) {
        cfg.token = Some(token);
    }

    if let Some(endpoint) = m.remove(GCS_SERVICE_PATH) {
        cfg.endpoint = Some(endpoint);
    }

    if m.remove(GCS_NO_AUTH).is_some() {
        cfg.allow_anonymous = true;
        cfg.disable_vm_metadata = true;
        cfg.disable_config_load = true;
    }

    if let Some(allow_anonymous) = m.remove(GCS_ALLOW_ANONYMOUS)
        && is_truthy(allow_anonymous.to_lowercase().as_str())
    {
        cfg.allow_anonymous = true;
    }
    if let Some(disable_ec2_metadata) = m.remove(GCS_DISABLE_VM_METADATA)
        && is_truthy(disable_ec2_metadata.to_lowercase().as_str())
    {
        cfg.disable_vm_metadata = true;
    };
    if let Some(disable_config_load) = m.remove(GCS_DISABLE_CONFIG_LOAD)
        && is_truthy(disable_config_load.to_lowercase().as_str())
    {
        cfg.disable_config_load = true;
    };

    Ok(cfg)
}

/// Build a new OpenDAL [`Operator`] based on a provided [`GcsConfig`].
pub(crate) fn gcs_config_build(cfg: &GcsConfig, path: &str) -> Result<Operator> {
    let url = Url::parse(path)?;
    let bucket = url.host_str().ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Invalid gcs url: {path}, bucket is required"),
        )
    })?;

    let mut cfg = cfg.clone();
    cfg.bucket = bucket.to_string();
    Ok(Operator::from_config(cfg)
        .map_err(from_opendal_error)?
        .finish())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use iceberg::io::{GCS_DISABLE_CONFIG_LOAD, GCS_DISABLE_VM_METADATA, GCS_NO_AUTH};

    use super::gcs_config_parse;
    use crate::{OpenDalStorage, OperatorCache};

    fn gcs_storage() -> OpenDalStorage {
        let props: HashMap<String, String> = [
            (GCS_NO_AUTH, "true"),
            (GCS_DISABLE_CONFIG_LOAD, "true"),
            (GCS_DISABLE_VM_METADATA, "true"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        OpenDalStorage::Gcs {
            config: Arc::new(gcs_config_parse(props).expect("offline gcs config parses")),
            operator_cache: OperatorCache::default(),
        }
    }

    #[test]
    fn test_create_operator_bucket_root_resolves_to_empty_key() {
        let storage = gcs_storage();
        let (op, rel) = storage
            .create_operator(&"gs://gcs-bucket")
            .expect("bare bucket root must resolve");
        assert_eq!(rel, "");
        assert_eq!(op.info().name(), "gcs-bucket");
        let (op2, rel2) = storage
            .create_operator(&"gs://gcs-bucket/")
            .expect("slash bucket root must resolve");
        assert_eq!(rel2, "");
        assert!(Arc::ptr_eq(op.inner(), op2.inner()));
        let (op3, rel3) = storage
            .create_operator(&"gs://gcs-bucket/nested/k")
            .expect("nested key must resolve");
        assert_eq!(rel3, "nested/k");
        assert!(Arc::ptr_eq(op.inner(), op3.inner()));
    }
}
