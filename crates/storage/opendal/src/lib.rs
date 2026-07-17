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

//! OpenDAL-based storage implementation for Apache Iceberg.
//!
//! This crate provides [`OpenDalStorage`] and [`OpenDalStorageFactory`],
//! which implement the [`Storage`](iceberg::io::Storage) and
//! [`StorageFactory`](iceberg::io::StorageFactory) traits from the `iceberg` crate
//! using [OpenDAL](https://opendal.apache.org/) as the backend.

mod utils;

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use cfg_if::cfg_if;
use iceberg::io::{
    FileInfo, FileMetadata, FileRead, FileWrite, InputFile, OutputFile, Storage, StorageConfig,
    StorageFactory,
};
use iceberg::{Error, ErrorKind, Result};
use opendal::Operator;
use opendal::layers::RetryLayer;
use serde::{Deserialize, Serialize};
use utils::from_opendal_error;

/// Convert an OpenDAL last-modified timestamp into milliseconds since the Unix epoch.
///
/// Mirrors how Java's object-store `FileIO` implementations populate `FileInfo.createdAtMillis`
/// from the object's last-modified time. Converts through `std::time::SystemTime` (an
/// infallible OpenDAL conversion) so no extra time-library dependency is needed. A timestamp at
/// or before the epoch clamps to `0` so the reported value stays non-negative.
fn opendal_timestamp_to_millis(timestamp: opendal::raw::Timestamp) -> i64 {
    let system_time: std::time::SystemTime = timestamp.into();
    match system_time.duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => i64::try_from(duration.as_millis()).unwrap_or(i64::MAX),
        Err(_) => 0,
    }
}

cfg_if! {
    if #[cfg(feature = "opendal-azdls")] {
        mod azdls;
        use azdls::AzureStorageScheme;
        use azdls::*;
        use opendal::services::AzdlsConfig;
    }
}

cfg_if! {
    if #[cfg(feature = "opendal-fs")] {
        mod fs;
        use fs::*;
    }
}

cfg_if! {
    if #[cfg(feature = "opendal-gcs")] {
        mod gcs;
        use gcs::*;
        use opendal::services::GcsConfig;
    }
}

cfg_if! {
    if #[cfg(feature = "opendal-memory")] {
        mod memory;
        use memory::*;
    }
}

cfg_if! {
    if #[cfg(feature = "opendal-oss")] {
        mod oss;
        use opendal::services::OssConfig;
        use oss::*;
    }
}

cfg_if! {
    if #[cfg(feature = "opendal-s3")] {
        mod s3;
        use opendal::services::S3Config;
        pub use s3::*;
    }
}

/// OpenDAL-based storage factory.
///
/// Maps scheme to the corresponding OpenDalStorage storage variant.
/// Use this factory with `FileIOBuilder::new(factory)` to create FileIO instances.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OpenDalStorageFactory {
    /// Memory storage factory.
    #[cfg(feature = "opendal-memory")]
    Memory,
    /// Local filesystem storage factory.
    #[cfg(feature = "opendal-fs")]
    Fs,
    /// S3 storage factory.
    #[cfg(feature = "opendal-s3")]
    S3 {
        /// s3 storage could have `s3://` and `s3a://`.
        /// Storing the scheme string here to return the correct path.
        configured_scheme: String,
        /// Custom AWS credential loader.
        #[serde(skip)]
        customized_credential_load: Option<s3::CustomAwsCredentialLoader>,
    },
    /// GCS storage factory.
    #[cfg(feature = "opendal-gcs")]
    Gcs,
    /// OSS storage factory.
    #[cfg(feature = "opendal-oss")]
    Oss,
    /// Azure Data Lake Storage factory.
    #[cfg(feature = "opendal-azdls")]
    Azdls {
        /// The configured Azure storage scheme.
        configured_scheme: AzureStorageScheme,
    },
}

#[typetag::serde(name = "OpenDalStorageFactory")]
impl StorageFactory for OpenDalStorageFactory {
    #[allow(unused_variables)]
    fn build(&self, config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        match self {
            #[cfg(feature = "opendal-memory")]
            OpenDalStorageFactory::Memory => {
                Ok(Arc::new(OpenDalStorage::Memory(memory_config_build()?)))
            }
            #[cfg(feature = "opendal-fs")]
            OpenDalStorageFactory::Fs => Ok(Arc::new(OpenDalStorage::LocalFs)),
            #[cfg(feature = "opendal-s3")]
            OpenDalStorageFactory::S3 {
                configured_scheme,
                customized_credential_load,
            } => Ok(Arc::new(OpenDalStorage::S3 {
                configured_scheme: configured_scheme.clone(),
                config: s3_config_parse(config.props().clone())?.into(),
                customized_credential_load: customized_credential_load.clone(),
            })),
            #[cfg(feature = "opendal-gcs")]
            OpenDalStorageFactory::Gcs => Ok(Arc::new(OpenDalStorage::Gcs {
                config: gcs_config_parse(config.props().clone())?.into(),
            })),
            #[cfg(feature = "opendal-oss")]
            OpenDalStorageFactory::Oss => Ok(Arc::new(OpenDalStorage::Oss {
                config: oss_config_parse(config.props().clone())?.into(),
            })),
            #[cfg(feature = "opendal-azdls")]
            OpenDalStorageFactory::Azdls { configured_scheme } => {
                Ok(Arc::new(OpenDalStorage::Azdls {
                    configured_scheme: configured_scheme.clone(),
                    config: azdls_config_parse(config.props().clone())?.into(),
                }))
            }
            #[cfg(all(
                not(feature = "opendal-memory"),
                not(feature = "opendal-fs"),
                not(feature = "opendal-s3"),
                not(feature = "opendal-gcs"),
                not(feature = "opendal-oss"),
                not(feature = "opendal-azdls"),
            ))]
            _ => Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "No storage service has been enabled",
            )),
        }
    }
}

/// Default memory operator for serde deserialization.
#[cfg(feature = "opendal-memory")]
fn default_memory_operator() -> Operator {
    memory_config_build().expect("Failed to create default memory operator")
}

/// OpenDAL-based storage implementation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OpenDalStorage {
    /// Memory storage variant.
    #[cfg(feature = "opendal-memory")]
    Memory(#[serde(skip, default = "self::default_memory_operator")] Operator),
    /// Local filesystem storage variant.
    #[cfg(feature = "opendal-fs")]
    LocalFs,
    /// S3 storage variant.
    #[cfg(feature = "opendal-s3")]
    S3 {
        /// s3 storage could have `s3://` and `s3a://`.
        /// Storing the scheme string here to return the correct path.
        configured_scheme: String,
        /// S3 configuration.
        config: Arc<S3Config>,
        /// Custom AWS credential loader.
        #[serde(skip)]
        customized_credential_load: Option<s3::CustomAwsCredentialLoader>,
    },
    /// GCS storage variant.
    #[cfg(feature = "opendal-gcs")]
    Gcs {
        /// GCS configuration.
        config: Arc<GcsConfig>,
    },
    /// OSS storage variant.
    #[cfg(feature = "opendal-oss")]
    Oss {
        /// OSS configuration.
        config: Arc<OssConfig>,
    },
    /// Azure Data Lake Storage variant.
    /// Expects paths of the form
    /// `abfs[s]://<filesystem>@<account>.dfs.<endpoint-suffix>/<path>` or
    /// `wasb[s]://<container>@<account>.blob.<endpoint-suffix>/<path>`.
    #[cfg(feature = "opendal-azdls")]
    #[allow(private_interfaces)]
    Azdls {
        /// The configured Azure storage scheme.
        /// Because Azdls accepts multiple possible schemes, we store the full
        /// passed scheme here to later validate schemes passed via paths.
        configured_scheme: AzureStorageScheme,
        /// Azure DLS configuration.
        config: Arc<AzdlsConfig>,
    },
}

impl OpenDalStorage {
    /// Creates operator from path.
    ///
    /// # Arguments
    ///
    /// * path: It should be *absolute* path starting with scheme string used to construct [`FileIO`](iceberg::io::FileIO).
    ///
    /// # Returns
    ///
    /// The return value consists of two parts:
    ///
    /// * An [`opendal::Operator`] instance used to operate on file.
    /// * Relative path to the root uri of [`opendal::Operator`].
    #[allow(unreachable_code, unused_variables)]
    pub(crate) fn create_operator<'a>(
        &self,
        path: &'a impl AsRef<str>,
    ) -> Result<(Operator, &'a str)> {
        let path = path.as_ref();
        let (operator, relative_path): (Operator, &str) = match self {
            #[cfg(feature = "opendal-memory")]
            OpenDalStorage::Memory(op) => {
                if let Some(stripped) = path.strip_prefix("memory:/") {
                    (op.clone(), stripped)
                } else {
                    (op.clone(), &path[1..])
                }
            }
            #[cfg(feature = "opendal-fs")]
            OpenDalStorage::LocalFs => {
                let op = fs_config_build()?;
                if let Some(stripped) = path.strip_prefix("file:/") {
                    (op, stripped)
                } else {
                    (op, &path[1..])
                }
            }
            #[cfg(feature = "opendal-s3")]
            OpenDalStorage::S3 {
                configured_scheme,
                config,
                customized_credential_load,
            } => {
                let op = s3_config_build(config, customized_credential_load, path)?;
                // `s3_config_build` derives the operator's bucket from `path`.
                let bucket = op.info().name().to_string();

                // `s3`, `s3a`, and `s3n` are aliases of the same object store
                // (Java `S3FileIO` parity): a location for this bucket resolves
                // under ANY alias, regardless of which alias the storage was
                // configured with. The relative key is stripped using the matched
                // alias's prefix length (see `s3_relative_path`).
                match s3_relative_path(path, &bucket) {
                    Some(relative_path) => (op, relative_path),
                    None => {
                        let accepted = S3_SCHEME_ALIASES
                            .iter()
                            .map(|&scheme| format!("{scheme}://{bucket}/"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Invalid s3 url: {path}, should start with one of \
                                 [{accepted}] (storage configured for scheme \
                                 {configured_scheme})"
                            ),
                        ));
                    }
                }
            }
            #[cfg(feature = "opendal-gcs")]
            OpenDalStorage::Gcs { config } => {
                let operator = gcs_config_build(config, path)?;
                let prefix = format!("gs://{}/", operator.info().name());
                if path.starts_with(&prefix) {
                    (operator, &path[prefix.len()..])
                } else {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid gcs url: {path}, should start with {prefix}"),
                    ));
                }
            }
            #[cfg(feature = "opendal-oss")]
            OpenDalStorage::Oss { config } => {
                let op = oss_config_build(config, path)?;
                let prefix = format!("oss://{}/", op.info().name());
                if path.starts_with(&prefix) {
                    (op, &path[prefix.len()..])
                } else {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid oss url: {path}, should start with {prefix}"),
                    ));
                }
            }
            #[cfg(feature = "opendal-azdls")]
            OpenDalStorage::Azdls {
                configured_scheme,
                config,
            } => azdls_create_operator(path, config, configured_scheme)?,
            #[cfg(all(
                not(feature = "opendal-s3"),
                not(feature = "opendal-fs"),
                not(feature = "opendal-gcs"),
                not(feature = "opendal-oss"),
                not(feature = "opendal-azdls"),
            ))]
            _ => {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    "No storage service has been enabled",
                ));
            }
        };

        // Transient errors are common for object stores; however there's no
        // harm in retrying temporary failures for other storage backends as well.
        let operator = operator.layer(RetryLayer::new());
        Ok((operator, relative_path))
    }
}

#[typetag::serde(name = "OpenDalStorage")]
#[async_trait]
impl Storage for OpenDalStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        let (op, relative_path) = self.create_operator(&path)?;
        Ok(op.exists(relative_path).await.map_err(from_opendal_error)?)
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        let (op, relative_path) = self.create_operator(&path)?;
        let meta = op.stat(relative_path).await.map_err(from_opendal_error)?;
        Ok(FileMetadata {
            size: meta.content_length(),
        })
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        let (op, relative_path) = self.create_operator(&path)?;
        Ok(op
            .read(relative_path)
            .await
            .map_err(from_opendal_error)?
            .to_bytes())
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        let (op, relative_path) = self.create_operator(&path)?;
        Ok(Box::new(OpenDalReader(
            op.reader(relative_path).await.map_err(from_opendal_error)?,
        )))
    }

    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        let (op, relative_path) = self.create_operator(&path)?;
        op.write(relative_path, bs)
            .await
            .map_err(from_opendal_error)?;
        Ok(())
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        let (op, relative_path) = self.create_operator(&path)?;
        Ok(Box::new(OpenDalWriter(
            op.writer(relative_path).await.map_err(from_opendal_error)?,
        )))
    }

    async fn delete(&self, path: &str) -> Result<()> {
        let (op, relative_path) = self.create_operator(&path)?;
        Ok(op.delete(relative_path).await.map_err(from_opendal_error)?)
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        let (op, relative_path) = self.create_operator(&path)?;
        let path = if relative_path.ends_with('/') {
            relative_path.to_string()
        } else {
            format!("{relative_path}/")
        };
        Ok(op.remove_all(&path).await.map_err(from_opendal_error)?)
    }

    /// Recursively list every file under `prefix`.
    ///
    /// # Prefix semantics: object-store / recursive
    ///
    /// OpenDAL's lister with `recursive(true)` walks every entry under the prefix, mirroring
    /// Java's object-store `FileIO` implementations (which list keys under a prefix) and the
    /// recursive `HadoopFileIO.listPrefix`. Only file entries are reported; directory markers
    /// are skipped. The prefix is normalized to a trailing-`/` directory boundary (the same
    /// shape `delete_prefix` removes), so a sibling key `ab2/...` is not reported for prefix
    /// `ab`.
    ///
    /// Each file's authoritative `size` and last-modified time are read via `stat` so the
    /// reported metadata is correct across every OpenDAL backend (some backends do not
    /// populate full metadata on a list). A file whose backend reports no last-modified time
    /// is reported with `created_at_millis = 0`.
    async fn list(&self, path: &str) -> Result<Vec<FileInfo>> {
        let (op, relative_path) = self.create_operator(&path)?;
        // The base is the part of the caller-supplied `path` that precedes the
        // operator-relative portion, so the entry's relative path can be re-prefixed back
        // into the scheme-qualified location the caller knows.
        let base = &path[..path.len() - relative_path.len()];

        let list_root = if relative_path.is_empty() || relative_path.ends_with('/') {
            relative_path.to_string()
        } else {
            format!("{relative_path}/")
        };

        let entries = op
            .list_with(&list_root)
            .recursive(true)
            .await
            .map_err(from_opendal_error)?;

        let mut files = Vec::with_capacity(entries.len());
        for entry in entries {
            if !entry.metadata().is_file() {
                continue;
            }
            let stat = op.stat(entry.path()).await.map_err(from_opendal_error)?;
            let created_at_millis = stat
                .last_modified()
                .map(opendal_timestamp_to_millis)
                .unwrap_or(0);
            files.push(FileInfo::new(
                format!("{base}{}", entry.path()),
                stat.content_length(),
                created_at_millis,
            ));
        }
        Ok(files)
    }

    #[allow(unreachable_code, unused_variables)]
    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    #[allow(unreachable_code, unused_variables)]
    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

// Newtype wrappers for opendal types to satisfy orphan rules.
// We can't implement iceberg's FileRead/FileWrite traits directly on opendal's
// Reader/Writer since neither trait nor type is defined in this crate.

/// Wrapper around `opendal::Reader` that implements `FileRead`.
pub(crate) struct OpenDalReader(pub(crate) opendal::Reader);

#[async_trait]
impl FileRead for OpenDalReader {
    async fn read(&self, range: std::ops::Range<u64>) -> Result<Bytes> {
        Ok(opendal::Reader::read(&self.0, range)
            .await
            .map_err(from_opendal_error)?
            .to_bytes())
    }
}

/// Wrapper around `opendal::Writer` that implements `FileWrite`.
pub(crate) struct OpenDalWriter(pub(crate) opendal::Writer);

#[async_trait]
impl FileWrite for OpenDalWriter {
    async fn write(&mut self, bs: Bytes) -> Result<()> {
        Ok(opendal::Writer::write(&mut self.0, bs)
            .await
            .map_err(from_opendal_error)?)
    }

    async fn close(&mut self) -> Result<()> {
        let _ = opendal::Writer::close(&mut self.0)
            .await
            .map_err(from_opendal_error)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "opendal-memory")]
    #[test]
    fn test_default_memory_operator() {
        let op = default_memory_operator();
        assert_eq!(op.info().scheme().to_string(), "memory");
    }

    /// Risk: the OpenDAL listing must return the exact recursive file set, with the right
    /// scheme-qualified locations and sizes, and must never report a sibling key outside the
    /// prefix (over-listing is over-deletion in the orphan-file action). Smoke test over the
    /// in-memory service.
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_opendal_memory_list_recursive_and_prefix_bounded() {
        let storage = OpenDalStorage::Memory(memory_config_build().unwrap());

        storage
            .write("memory:/dir/a.txt", Bytes::from("a"))
            .await
            .unwrap();
        storage
            .write("memory:/dir/sub/b.txt", Bytes::from("bb"))
            .await
            .unwrap();
        // A sibling key under a different prefix that must NOT appear.
        storage
            .write("memory:/dir2/c.txt", Bytes::from("ccc"))
            .await
            .unwrap();

        let mut listed = storage.list("memory:/dir").await.unwrap();
        listed.sort_by(|left, right| left.location.cmp(&right.location));

        let locations: Vec<&str> = listed.iter().map(|f| f.location.as_str()).collect();
        assert_eq!(locations, vec![
            "memory:/dir/a.txt",
            "memory:/dir/sub/b.txt"
        ]);
        assert!(!locations.contains(&"memory:/dir2/c.txt"));

        let by_location = |location: &str| listed.iter().find(|f| f.location == location).unwrap();
        assert_eq!(by_location("memory:/dir/a.txt").size, 1);
        assert_eq!(by_location("memory:/dir/sub/b.txt").size, 2);
    }

    /// Risk: a prefix with no matching files must be a legitimate empty answer, not an error
    /// and not a stale/over-broad listing.
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_opendal_memory_list_empty_prefix_is_empty() {
        let storage = OpenDalStorage::Memory(memory_config_build().unwrap());
        storage
            .write("memory:/other/a.txt", Bytes::from("a"))
            .await
            .unwrap();

        let listed = storage.list("memory:/nothing-here").await.unwrap();
        assert!(listed.is_empty());
    }

    /// Risk: a wrong epoch base or a secs/millis mix-up in the OpenDAL last-modified conversion
    /// would feed A2 nonsense timestamps. Pins the conversion at exact boundaries: epoch -> 0,
    /// 1 ms -> 1, a pre-epoch instant clamps to 0 (never negative), and a known recent
    /// millisecond value round-trips exactly (proving milliseconds-since-epoch, not seconds).
    #[test]
    fn test_opendal_timestamp_conversion_is_exact_milliseconds_and_clamps_pre_epoch() {
        let epoch = opendal::raw::Timestamp::from_millisecond(0).unwrap();
        assert_eq!(opendal_timestamp_to_millis(epoch), 0);

        let one_milli = opendal::raw::Timestamp::from_millisecond(1).unwrap();
        assert_eq!(opendal_timestamp_to_millis(one_milli), 1);

        let pre_epoch = opendal::raw::Timestamp::from_millisecond(-1000).unwrap();
        assert_eq!(
            opendal_timestamp_to_millis(pre_epoch),
            0,
            "a pre-epoch timestamp must clamp to 0, never produce a negative value"
        );

        let known_millis: i64 = 1_700_000_000_000;
        let known = opendal::raw::Timestamp::from_millisecond(known_millis).unwrap();
        assert_eq!(
            opendal_timestamp_to_millis(known),
            known_millis,
            "a known recent value must round-trip exactly as milliseconds-since-epoch"
        );
    }

    /// S3 scheme aliasing (F-A2-1): `s3`/`s3a`/`s3n` are aliases of the same
    /// storage (Java `S3FileIO` parity). Every pin builds an `OpenDalStorage::S3`
    /// offline — a fixed region with ambient config/EC2 loads disabled — so the
    /// opendal operator is constructed without any AWS contact.
    #[cfg(feature = "opendal-s3")]
    mod s3_scheme_alias {
        use std::sync::Arc;

        use iceberg::io::{
            S3_DISABLE_CONFIG_LOAD, S3_DISABLE_EC2_METADATA, S3_REGION, StorageConfig,
            StorageFactory,
        };

        use crate::{OpenDalStorage, OpenDalStorageFactory};

        /// Offline S3 props: a fixed region plus disabled ambient config/EC2 loads
        /// so the operator builds without any network or credential probe.
        fn offline_s3_props() -> Vec<(&'static str, &'static str)> {
            vec![
                (S3_REGION, "us-east-1"),
                (S3_DISABLE_CONFIG_LOAD, "true"),
                (S3_DISABLE_EC2_METADATA, "true"),
            ]
        }

        /// Build an `OpenDalStorage::S3` exactly as `OpenDalStorageFactory::S3`
        /// does, for the given configured scheme. Offline: no AWS contact.
        fn s3_storage(configured_scheme: &str) -> OpenDalStorage {
            let props = offline_s3_props()
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            let config = crate::s3::s3_config_parse(props).expect("offline s3 config parses");
            OpenDalStorage::S3 {
                configured_scheme: configured_scheme.to_string(),
                config: Arc::new(config),
                customized_credential_load: None,
            }
        }

        /// Assert a location resolves against a store configured with `configured`,
        /// pinning both the operator's bucket and the exact operator-relative key.
        fn assert_resolves(configured: &str, path: &str, bucket: &str, key: &str) {
            let storage = s3_storage(configured);
            let (op, relative_path) = storage
                .create_operator(&path)
                .unwrap_or_else(|e| panic!("{path} must resolve for configured {configured}: {e}"));
            assert_eq!(relative_path, key, "relative key for {path}");
            assert_eq!(op.info().name(), bucket, "bucket for {path}");
        }

        /// Element 1: the Glue default (configured `s3a`) accepts a canonical
        /// `s3://` location — the exact acceptance-run failure being fixed.
        #[test]
        fn test_create_operator_configured_s3a_accepts_s3_scheme() {
            assert_resolves("s3a", "s3://my-bucket/k", "my-bucket", "k");
        }

        /// Element 2 (regression): configured `s3a` still accepts `s3a://`.
        #[test]
        fn test_create_operator_configured_s3a_accepts_s3a_scheme_regression() {
            assert_resolves("s3a", "s3a://my-bucket/k", "my-bucket", "k");
        }

        /// Element 3: configured `s3` accepts `s3a://`.
        #[test]
        fn test_create_operator_configured_s3_accepts_s3a_scheme() {
            assert_resolves("s3", "s3a://my-bucket/k", "my-bucket", "k");
        }

        /// Element 4 (regression): configured `s3` still accepts `s3://`.
        #[test]
        fn test_create_operator_configured_s3_accepts_s3_scheme_regression() {
            assert_resolves("s3", "s3://my-bucket/k", "my-bucket", "k");
        }

        /// Element 5: the `s3n` alias resolves too.
        #[test]
        fn test_create_operator_accepts_s3n_scheme() {
            assert_resolves("s3a", "s3n://my-bucket/k", "my-bucket", "k");
        }

        /// Element 7: non-S3 schemes stay rejected by the S3 arm.
        #[test]
        fn test_create_operator_rejects_non_s3_schemes() {
            let storage = s3_storage("s3a");
            for path in ["gs://my-bucket/k", "file:///tmp/k", "my-bucket/k"] {
                assert!(
                    storage.create_operator(&path).is_err(),
                    "{path} must be rejected by the S3 arm"
                );
            }
            // A well-formed non-S3 scheme reaches the alias check; the error names
            // the rejected location.
            let err = storage
                .create_operator(&"gs://my-bucket/k")
                .expect_err("gs:// must be rejected");
            assert!(
                err.to_string().contains("gs://my-bucket/k"),
                "error must name the rejected location, got: {err}"
            );
        }

        /// Element 9 (end-to-end): the Glue catalog's default FileIO factory
        /// (`configured_scheme: "s3a"`) composes with a real `s3://` metadata
        /// location. Proves the catalog default + canonical metadata locations now
        /// resolve together at the single funnel every `Storage` I/O routes through.
        #[test]
        fn test_glue_default_factory_composes_with_s3_metadata_location() {
            // The Glue default, built through the real factory + StorageConfig path.
            let factory = OpenDalStorageFactory::S3 {
                configured_scheme: "s3a".to_string(),
                customized_credential_load: None,
            };
            let config = StorageConfig::new().with_props(offline_s3_props());
            let _built = factory
                .build(&config)
                .expect("Glue-default S3 factory must build from Glue-shaped props");

            // `factory.build` yields an `Arc<dyn Storage>` (create_operator is not on
            // the trait); the concrete store below is byte-identical to what the S3
            // factory arm constructs, so the location is resolved on it.
            let storage = s3_storage("s3a");
            let location = "s3://warehouse-bucket/db/tbl/metadata/00001-1a2b-uuid.metadata.json";
            let (op, relative_path) = storage
                .create_operator(&location)
                .expect("a real s3:// metadata location must compose with the s3a Glue default");
            assert_eq!(
                relative_path,
                "db/tbl/metadata/00001-1a2b-uuid.metadata.json"
            );
            assert_eq!(op.info().name(), "warehouse-bucket");
            assert_eq!(op.info().scheme().to_string(), "s3");
        }
    }
}
