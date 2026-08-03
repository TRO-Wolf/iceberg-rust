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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use cfg_if::cfg_if;
use iceberg::io::{
    CLIENT_LIST_STAT_CONCURRENCY, DEFAULT_LIST_STAT_CONCURRENCY, FileInfo, FileMetadata, FileRead,
    FileWrite, InputFile, OutputFile, Storage, StorageConfig, StorageFactory,
};
use iceberg::{Error, ErrorKind, Result};
use opendal::layers::{ConcurrentLimitLayer, RetryLayer};
use opendal::raw::ConcurrentTasks;
use opendal::{Executor, Operator};
use serde::{Deserialize, Serialize};
use utils::from_opendal_error;

/// Per-operator concurrent request cap applied once when an Operator is first cached.
///
/// Each cached Operator gets its own [`ConcurrentLimitLayer`] (independent semaphore).
/// Chosen as a conservative default; not a global process-wide limit.
const OPERATOR_CONCURRENT_LIMIT: usize = 64;

/// Parse [`CLIENT_LIST_STAT_CONCURRENCY`] from FileIO props.
///
/// Missing / unparseable → [`DEFAULT_LIST_STAT_CONCURRENCY`] (16). `0` clamps to `1`.
fn parse_list_stat_concurrency(props: &HashMap<String, String>) -> usize {
    match props.get(CLIENT_LIST_STAT_CONCURRENCY) {
        Some(raw) => match raw.parse::<usize>() {
            Ok(0) => 1,
            Ok(n) => n,
            Err(_) => DEFAULT_LIST_STAT_CONCURRENCY,
        },
        None => DEFAULT_LIST_STAT_CONCURRENCY,
    }
}

/// Convert OpenDAL [`opendal::Buffer`] to contiguous [`Bytes`].
///
/// Prefers the zero-copy path: [`opendal::Buffer::to_bytes`] clones the inner
/// `Bytes` when the buffer is already contiguous (or a single part). Multi-part
/// non-contiguous buffers must consolidate into one `Bytes` — unavoidable for
/// the Iceberg `FileRead` / `Storage::read` API, which returns a single
/// contiguous buffer.
#[inline]
fn buffer_to_bytes(buf: opendal::Buffer) -> Bytes {
    buf.to_bytes()
}

/// Input to a concurrent list-`stat` task: (operator, relative path, slot index).
type ListStatIn = (Operator, String, usize);
/// Output of a concurrent list-`stat` task: (slot index, size, created_at_millis).
type ListStatOut = (usize, u64, i64);

/// Factory for [`ConcurrentTasks`] — must be a function pointer (no captures).
fn list_stat_task(
    input: ListStatIn,
) -> opendal::raw::BoxedStaticFuture<(ListStatIn, opendal::Result<ListStatOut>)> {
    Box::pin(async move {
        let (op, path, slot_idx) = input;
        let result = match op.stat(&path).await {
            Ok(meta) => Ok((
                slot_idx,
                meta.content_length(),
                meta.last_modified()
                    .map(opendal_timestamp_to_millis)
                    .unwrap_or(0),
            )),
            Err(err) => Err(err),
        };
        ((op, path, slot_idx), result)
    })
}

/// Run `stat` for incomplete list entries with a bounded concurrency window.
///
/// Results are applied into `ready_meta[slot_idx]`. Order of the parent list is
/// preserved by slot index (not by completion order).
async fn stat_incomplete_list_entries(
    op: &Operator,
    need_stat: &[(usize, String)],
    concurrency: usize,
    ready_meta: &mut [Option<(u64, i64)>],
) -> Result<()> {
    if need_stat.is_empty() {
        return Ok(());
    }
    let concurrency = concurrency.max(1);
    let mut tasks = ConcurrentTasks::new(Executor::new(), concurrency, concurrency, list_stat_task);
    // Fail-closed: any stat error fails the whole list. OpenDAL's RetryLayer on
    // the Operator already retried transport blips; do **not** outer-loop on
    // `is_temporary` here — ConcurrentTasks re-queues temporary failures and an
    // unbounded continue would hang orphan/GC list forever (C1-Q-001 / C1-L-001).
    for &(slot_idx, ref path) in need_stat {
        tasks
            .execute((op.clone(), path.clone(), slot_idx))
            .await
            .map_err(from_opendal_error)?;
    }
    loop {
        match tasks.next().await {
            None => break,
            Some(Ok((slot_idx, size, created_at_millis))) => {
                let slot = ready_meta.get_mut(slot_idx).ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!("list stat returned out-of-range slot index {slot_idx}"),
                    )
                })?;
                *slot = Some((size, created_at_millis));
            }
            Some(Err(err)) => return Err(from_opendal_error(err)),
        }
    }
    Ok(())
}

/// Apply transport layers once at Operator construction / cache insertion.
///
/// Layers are **not** re-applied on cache hits — stacking `RetryLayer` on every
/// `create_operator` call would multiply retries and allocate a new Operator wrapper
/// per I/O.
fn finish_operator(op: Operator) -> Operator {
    op.layer(RetryLayer::new())
        .layer(ConcurrentLimitLayer::new(OPERATOR_CONCURRENT_LIMIT))
}

/// Thread-safe cache of finished OpenDAL [`Operator`]s, keyed by backend name
/// (S3/GCS/OSS bucket, AzDLS filesystem, or a fixed key for FS / Memory).
///
/// Cloning shares the map via [`Arc`] so `OpenDalStorage` clones (e.g. per
/// `InputFile` / `OutputFile`) reuse Operators.
///
/// Also carries list-path tuning ([`OperatorCache::list_stat_concurrency`]) set
/// once at storage construction from FileIO props — every [`OpenDalStorage`]
/// variant holds a cache, so this avoids duplicating the knob on each arm.
///
/// Public only because it appears on [`OpenDalStorage`] enum fields (serde +
/// construction); the map itself is an implementation detail.
#[derive(Clone)]
#[doc(hidden)]
pub struct OperatorCache {
    inner: Arc<Mutex<HashMap<String, Operator>>>,
    /// Max concurrent `stat` HEADs for incomplete list entries (FK4.2).
    list_stat_concurrency: usize,
}

impl Default for OperatorCache {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            list_stat_concurrency: DEFAULT_LIST_STAT_CONCURRENCY,
        }
    }
}

impl std::fmt::Debug for OperatorCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.inner.lock().map(|g| g.len()).unwrap_or(0);
        f.debug_struct("OperatorCache")
            .field("entries", &len)
            .field("list_stat_concurrency", &self.list_stat_concurrency)
            .finish()
    }
}

impl OperatorCache {
    /// Set list incomplete-entry `stat` concurrency (clamped to at least 1).
    fn with_list_stat_concurrency(mut self, n: usize) -> Self {
        self.list_stat_concurrency = n.max(1);
        self
    }

    /// Return a cached finished Operator for `key`, building and finishing it on miss.
    fn get_or_insert_with(
        &self,
        key: String,
        build: impl FnOnce() -> Result<Operator>,
    ) -> Result<Operator> {
        {
            let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(op) = guard.get(&key) {
                return Ok(op.clone());
            }
        }

        // Build under the lock so concurrent first-accesses for the same key do not
        // each construct an Operator (connection pools / HTTP clients).
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(op) = guard.get(&key) {
            return Ok(op.clone());
        }
        let op = finish_operator(build()?);
        guard.insert(key, op.clone());
        Ok(op)
    }
}

/// Build an [`OperatorCache`] from FileIO [`StorageConfig`] props (list-stat knob).
fn operator_cache_from_config(config: &StorageConfig) -> OperatorCache {
    OperatorCache::default().with_list_stat_concurrency(parse_list_stat_concurrency(config.props()))
}

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

/// Whether list-entry metadata is complete enough to skip a per-file `stat`.
///
/// **Rule:** use list metadata only when the backend reported a **positive**
/// `content_length`. OpenDAL's public `content_length()` returns `0` both when
/// size was never set (`None`) and when the object is legitimately empty
/// (`Some(0)`), so a zero length is **never** treated as complete — empty and
/// unknown entries always fall back to `stat` (one HEAD; correct size 0).
///
/// Do **not** treat `last_modified` alone as proof of size: backends can set
/// mtime without size (e.g. S3/OSS delete markers under `list_with_deleted`),
/// which would otherwise be reported as empty files without `stat`.
///
/// Some backends (e.g. in-memory OpenDAL, local FS list) never populate list
/// size — those paths always `stat`. Object-store LIST responses that carry a
/// non-zero Size skip the N HEAD round-trips for real data files.
fn list_entry_metadata_complete(meta: &opendal::Metadata) -> bool {
    !meta.is_deleted() && meta.content_length() > 0
}

/// Size + created-at millis taken from a **complete** list entry (no `stat`).
///
/// Call only when [`list_entry_metadata_complete`] is true. Missing
/// `last_modified` becomes `created_at_millis = 0`.
fn file_meta_from_complete_list_entry(meta: &opendal::Metadata) -> (u64, i64) {
    let size = meta.content_length();
    let created_at_millis = meta
        .last_modified()
        .map(opendal_timestamp_to_millis)
        .unwrap_or(0);
    (size, created_at_millis)
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
        let operator_cache = operator_cache_from_config(config);
        match self {
            #[cfg(feature = "opendal-memory")]
            OpenDalStorageFactory::Memory => Ok(Arc::new(OpenDalStorage::Memory {
                operator: memory_config_build()?,
                operator_cache,
            })),
            #[cfg(feature = "opendal-fs")]
            OpenDalStorageFactory::Fs => Ok(Arc::new(OpenDalStorage::LocalFs { operator_cache })),
            #[cfg(feature = "opendal-s3")]
            OpenDalStorageFactory::S3 {
                configured_scheme,
                customized_credential_load,
            } => Ok(Arc::new(OpenDalStorage::S3 {
                configured_scheme: configured_scheme.clone(),
                config: s3_config_parse(config.props().clone())?.into(),
                customized_credential_load: customized_credential_load.clone(),
                operator_cache,
            })),
            #[cfg(feature = "opendal-gcs")]
            OpenDalStorageFactory::Gcs => Ok(Arc::new(OpenDalStorage::Gcs {
                config: gcs_config_parse(config.props().clone())?.into(),
                operator_cache,
            })),
            #[cfg(feature = "opendal-oss")]
            OpenDalStorageFactory::Oss => Ok(Arc::new(OpenDalStorage::Oss {
                config: oss_config_parse(config.props().clone())?.into(),
                operator_cache,
            })),
            #[cfg(feature = "opendal-azdls")]
            OpenDalStorageFactory::Azdls { configured_scheme } => {
                Ok(Arc::new(OpenDalStorage::Azdls {
                    configured_scheme: configured_scheme.clone(),
                    config: azdls_config_parse(config.props().clone())?.into(),
                    operator_cache,
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
    Memory {
        /// Underlying OpenDAL memory operator (raw; layers applied via cache).
        #[serde(skip, default = "self::default_memory_operator")]
        operator: Operator,
        /// Finished-operator cache (single entry keyed `"memory"`). Shared on clone.
        #[serde(skip, default)]
        operator_cache: OperatorCache,
    },
    /// Local filesystem storage variant.
    #[cfg(feature = "opendal-fs")]
    LocalFs {
        /// Finished-operator cache (single entry keyed `"fs"`). Shared on clone.
        #[serde(skip, default)]
        operator_cache: OperatorCache,
    },
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
        /// Operators keyed by bucket name. Shared on clone.
        #[serde(skip, default)]
        operator_cache: OperatorCache,
    },
    /// GCS storage variant.
    #[cfg(feature = "opendal-gcs")]
    Gcs {
        /// GCS configuration.
        config: Arc<GcsConfig>,
        /// Operators keyed by bucket name. Shared on clone.
        #[serde(skip, default)]
        operator_cache: OperatorCache,
    },
    /// OSS storage variant.
    #[cfg(feature = "opendal-oss")]
    Oss {
        /// OSS configuration.
        config: Arc<OssConfig>,
        /// Operators keyed by bucket name. Shared on clone.
        #[serde(skip, default)]
        operator_cache: OperatorCache,
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
        /// Operators keyed by filesystem name. Shared on clone.
        #[serde(skip, default)]
        operator_cache: OperatorCache,
    },
}

impl OpenDalStorage {
    /// List incomplete-entry `stat` concurrency for this storage instance.
    fn list_stat_concurrency(&self) -> usize {
        match self {
            #[cfg(feature = "opendal-memory")]
            OpenDalStorage::Memory { operator_cache, .. } => operator_cache.list_stat_concurrency,
            #[cfg(feature = "opendal-fs")]
            OpenDalStorage::LocalFs { operator_cache } => operator_cache.list_stat_concurrency,
            #[cfg(feature = "opendal-s3")]
            OpenDalStorage::S3 { operator_cache, .. } => operator_cache.list_stat_concurrency,
            #[cfg(feature = "opendal-gcs")]
            OpenDalStorage::Gcs { operator_cache, .. } => operator_cache.list_stat_concurrency,
            #[cfg(feature = "opendal-oss")]
            OpenDalStorage::Oss { operator_cache, .. } => operator_cache.list_stat_concurrency,
            #[cfg(feature = "opendal-azdls")]
            OpenDalStorage::Azdls { operator_cache, .. } => operator_cache.list_stat_concurrency,
            #[cfg(all(
                not(feature = "opendal-memory"),
                not(feature = "opendal-fs"),
                not(feature = "opendal-s3"),
                not(feature = "opendal-gcs"),
                not(feature = "opendal-oss"),
                not(feature = "opendal-azdls"),
            ))]
            _ => DEFAULT_LIST_STAT_CONCURRENCY,
        }
    }

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
        match self {
            #[cfg(feature = "opendal-memory")]
            OpenDalStorage::Memory {
                operator,
                operator_cache,
            } => {
                let relative_path = if let Some(stripped) = path.strip_prefix("memory:/") {
                    stripped
                } else {
                    &path[1..]
                };
                let op = operator_cache
                    .get_or_insert_with("memory".to_string(), || Ok(operator.clone()))?;
                Ok((op, relative_path))
            }
            #[cfg(feature = "opendal-fs")]
            OpenDalStorage::LocalFs { operator_cache } => {
                let relative_path = if let Some(stripped) = path.strip_prefix("file:/") {
                    stripped
                } else {
                    &path[1..]
                };
                let op = operator_cache.get_or_insert_with("fs".to_string(), fs_config_build)?;
                Ok((op, relative_path))
            }
            #[cfg(feature = "opendal-s3")]
            OpenDalStorage::S3 {
                configured_scheme,
                config,
                customized_credential_load,
                operator_cache,
            } => {
                // Derive the bucket from the URL host without building an Operator,
                // so cache hits skip `s3_config_build` entirely.
                let url = url::Url::parse(path).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid s3 url: {path}: {e}"),
                    )
                })?;
                let bucket = url.host_str().ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid s3 url: {path}, missing bucket"),
                    )
                })?;

                // `s3`, `s3a`, and `s3n` are aliases of the same object store
                // (Java `S3FileIO` parity): a location for this bucket resolves
                // under ANY alias, regardless of which alias the storage was
                // configured with. The relative key is stripped using the matched
                // alias's prefix length (see `s3_relative_path`).
                let relative_path = match s3_relative_path(path, bucket) {
                    Some(relative_path) => relative_path,
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
                };

                let op = operator_cache.get_or_insert_with(bucket.to_string(), || {
                    s3_config_build(config, customized_credential_load, path)
                })?;
                Ok((op, relative_path))
            }
            #[cfg(feature = "opendal-gcs")]
            OpenDalStorage::Gcs {
                config,
                operator_cache,
            } => {
                let url = url::Url::parse(path).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid gcs url: {path}: {e}"),
                    )
                })?;
                let bucket = url.host_str().ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid gcs url: {path}, bucket is required"),
                    )
                })?;
                let prefix = format!("gs://{bucket}/");
                if !path.starts_with(&prefix) {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid gcs url: {path}, should start with {prefix}"),
                    ));
                }
                let relative_path = &path[prefix.len()..];
                let op = operator_cache
                    .get_or_insert_with(bucket.to_string(), || gcs_config_build(config, path))?;
                Ok((op, relative_path))
            }
            #[cfg(feature = "opendal-oss")]
            OpenDalStorage::Oss {
                config,
                operator_cache,
            } => {
                let url = url::Url::parse(path).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid oss url: {path}: {e}"),
                    )
                })?;
                let bucket = url.host_str().ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid oss url: {path}, missing bucket"),
                    )
                })?;
                let prefix = format!("oss://{bucket}/");
                if !path.starts_with(&prefix) {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid oss url: {path}, should start with {prefix}"),
                    ));
                }
                let relative_path = &path[prefix.len()..];
                let op = operator_cache
                    .get_or_insert_with(bucket.to_string(), || oss_config_build(config, path))?;
                Ok((op, relative_path))
            }
            #[cfg(feature = "opendal-azdls")]
            OpenDalStorage::Azdls {
                configured_scheme,
                config,
                operator_cache,
            } => {
                // Parse/validate first so cache hits never construct an Operator.
                // Key by filesystem name (OpenDAL `info().name()`); build only on miss.
                let resolved = azdls_resolve(path, config, configured_scheme)?;
                let relative_path = resolved.relative_path;
                let op = operator_cache.get_or_insert_with(resolved.filesystem.clone(), || {
                    resolved.build_operator(config)
                })?;
                Ok((op, relative_path))
            }
            #[cfg(all(
                not(feature = "opendal-s3"),
                not(feature = "opendal-fs"),
                not(feature = "opendal-gcs"),
                not(feature = "opendal-oss"),
                not(feature = "opendal-azdls"),
                not(feature = "opendal-memory"),
            ))]
            _ => Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "No storage service has been enabled",
            )),
        }
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
        Ok(buffer_to_bytes(
            op.read(relative_path).await.map_err(from_opendal_error)?,
        ))
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
    /// # Metadata source
    ///
    /// Prefer size + last-modified from the list entry when
    /// [`list_entry_metadata_complete`] (positive `content_length`, not deleted).
    /// Only `stat` when list metadata is incomplete (zero/unknown size, or a
    /// deleted marker). Incomplete backends (e.g. OpenDAL memory, FS list) pay
    /// one `stat` per file; object-store LIST responses that already carry a
    /// non-zero Size skip the N HEAD round-trips for data files. Empty objects
    /// always `stat` so size 0 is authoritative. A file with no last-modified
    /// is reported with `created_at_millis = 0`.
    ///
    /// Incomplete-entry `stat`s run concurrently with a bound of
    /// [`CLIENT_LIST_STAT_CONCURRENCY`] (default
    /// [`DEFAULT_LIST_STAT_CONCURRENCY`] = 16). Raising the knob amplifies HEAD
    /// QPS and can hit object-store rate limits; the outer
    /// [`ConcurrentLimitLayer`] (64) still caps total in-flight ops per Operator.
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

        // Slot-oriented pass: complete list meta is ready immediately; incomplete
        // entries collect a concurrent `stat` job keyed by slot index so order is
        // preserved regardless of HEAD completion order.
        let mut locations: Vec<String> = Vec::with_capacity(entries.len());
        let mut ready_meta: Vec<Option<(u64, i64)>> = Vec::with_capacity(entries.len());
        let mut need_stat: Vec<(usize, String)> = Vec::new();

        for entry in entries {
            let list_meta = entry.metadata();
            // Skip directory markers and delete-marker entries (not live files).
            if !list_meta.is_file() || list_meta.is_deleted() {
                continue;
            }

            let location = format!("{base}{}", entry.path());
            if list_entry_metadata_complete(list_meta) {
                locations.push(location);
                ready_meta.push(Some(file_meta_from_complete_list_entry(list_meta)));
            } else {
                let slot_idx = locations.len();
                need_stat.push((slot_idx, entry.path().to_string()));
                locations.push(location);
                ready_meta.push(None);
            }
        }

        stat_incomplete_list_entries(
            &op,
            &need_stat,
            self.list_stat_concurrency(),
            &mut ready_meta,
        )
        .await?;

        let mut files = Vec::with_capacity(locations.len());
        for (location, meta) in locations.into_iter().zip(ready_meta.into_iter()) {
            let (size, created_at_millis) = meta.ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("list stat did not produce metadata for {location}"),
                )
            })?;
            files.push(FileInfo::new(location, size, created_at_millis));
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
        Ok(buffer_to_bytes(
            opendal::Reader::read(&self.0, range)
                .await
                .map_err(from_opendal_error)?,
        ))
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

    /// Helper: build an in-memory OpenDalStorage with a fresh operator + cache.
    #[cfg(feature = "opendal-memory")]
    fn memory_storage() -> OpenDalStorage {
        OpenDalStorage::Memory {
            operator: memory_config_build().expect("memory operator builds"),
            operator_cache: OperatorCache::default(),
        }
    }

    /// Risk: the OpenDAL listing must return the exact recursive file set, with the right
    /// scheme-qualified locations and sizes, and must never report a sibling key outside the
    /// prefix (over-listing is over-deletion in the orphan-file action). Smoke test over the
    /// in-memory service.
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_opendal_memory_list_recursive_and_prefix_bounded() {
        let storage = memory_storage();

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
        let storage = memory_storage();
        storage
            .write("memory:/other/a.txt", Bytes::from("a"))
            .await
            .unwrap();

        let listed = storage.list("memory:/nothing-here").await.unwrap();
        assert!(listed.is_empty());
    }

    /// Pins the list-metadata completeness rule used to skip `stat`.
    ///
    /// OpenDAL's public `content_length()` is 0 when unset **and** when the object
    /// is empty, so only a **positive** size is complete. Mtime alone must not
    /// skip stat (would report size 0 for unset length). Deleted markers are
    /// never complete.
    #[test]
    fn test_list_entry_metadata_complete_rule() {
        use opendal::{EntryMode, Metadata};

        let incomplete = Metadata::new(EntryMode::FILE);
        assert!(
            !list_entry_metadata_complete(&incomplete),
            "list entries with no size must fall back to stat"
        );

        let with_size = Metadata::new(EntryMode::FILE).with_content_length(42);
        assert!(
            list_entry_metadata_complete(&with_size),
            "positive content_length is complete"
        );

        let mtime_only = Metadata::new(EntryMode::FILE).with_last_modified(
            opendal::raw::Timestamp::from_millisecond(1_700_000_000_000).expect("ts"),
        );
        assert!(
            !list_entry_metadata_complete(&mtime_only),
            "mtime alone must not be treated as complete (unset size collapses to 0)"
        );

        let empty_with_mtime = Metadata::new(EntryMode::FILE)
            .with_content_length(0)
            .with_last_modified(
                opendal::raw::Timestamp::from_millisecond(1_700_000_000_000).expect("ts"),
            );
        assert!(
            !list_entry_metadata_complete(&empty_with_mtime),
            "empty objects (size 0) must stat so size is authoritative"
        );

        let deleted = Metadata::new(EntryMode::FILE)
            .with_content_length(42)
            .with_is_deleted(true);
        assert!(
            !list_entry_metadata_complete(&deleted),
            "delete markers must not be treated as complete live files"
        );
    }

    /// AzDLS: same filesystem reuses the cached finished Operator; a second
    /// filesystem gets a distinct Operator. Offline (no Azure network I/O).
    #[cfg(feature = "opendal-azdls")]
    #[test]
    fn test_operator_cache_reuses_operator_for_same_azdls_filesystem() {
        use opendal::services::AzdlsConfig;

        let config = AzdlsConfig {
            account_name: Some("myaccount".to_string()),
            endpoint: Some("https://myaccount.dfs.core.windows.net".to_string()),
            ..Default::default()
        };
        let storage = OpenDalStorage::Azdls {
            configured_scheme: AzureStorageScheme::Abfss,
            config: std::sync::Arc::new(config),
            operator_cache: OperatorCache::default(),
        };

        let p1 = "abfss://myfs@myaccount.dfs.core.windows.net/path/to/a.parquet";
        let p2 = "abfss://myfs@myaccount.dfs.core.windows.net/other/b.parquet";
        let p_other = "abfss://otherfs@myaccount.dfs.core.windows.net/x.parquet";

        let (op1, rel1) = storage.create_operator(&p1).expect("azdls first");
        let (op2, rel2) = storage.create_operator(&p2).expect("azdls second same fs");
        assert_eq!(rel1, "/path/to/a.parquet");
        assert_eq!(rel2, "/other/b.parquet");
        assert_eq!(op1.info().name(), "myfs");
        assert!(
            std::sync::Arc::ptr_eq(op1.inner(), op2.inner()),
            "same AzDLS filesystem must reuse the cached Operator"
        );

        let (op_other, rel_other) = storage
            .create_operator(&p_other)
            .expect("azdls other filesystem");
        assert_eq!(rel_other, "/x.parquet");
        assert_eq!(op_other.info().name(), "otherfs");
        assert!(
            !std::sync::Arc::ptr_eq(op1.inner(), op_other.inner()),
            "different filesystems must not share Operators"
        );
    }

    /// `delete_prefix` via cached Operator removes the prefix tree only.
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_memory_delete_prefix_through_operator_cache() {
        let storage = memory_storage();
        storage
            .write("memory:/pfx/a", Bytes::from("a"))
            .await
            .expect("write a");
        storage
            .write("memory:/pfx/sub/b", Bytes::from("b"))
            .await
            .expect("write b");
        storage
            .write("memory:/pfx2/c", Bytes::from("c"))
            .await
            .expect("write sibling");
        storage
            .delete_prefix("memory:/pfx")
            .await
            .expect("delete_prefix");
        assert!(!storage.exists("memory:/pfx/a").await.expect("a"));
        assert!(!storage.exists("memory:/pfx/sub/b").await.expect("b"));
        assert!(
            storage.exists("memory:/pfx2/c").await.expect("sibling"),
            "delete_prefix must not remove sibling prefix"
        );
    }

    /// Streaming writer through cached Operator flushes on close.
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_memory_writer_close_through_operator_cache() {
        let storage = memory_storage();
        let mut w = storage.writer("memory:/w/out.bin").await.expect("writer");
        w.write(Bytes::from("hello")).await.expect("write chunk");
        w.write(Bytes::from(" world")).await.expect("write chunk 2");
        w.close().await.expect("close");
        let body = storage
            .read("memory:/w/out.bin")
            .await
            .expect("read after close");
        assert_eq!(body.as_ref(), b"hello world");
    }

    /// Delete through the cached Operator removes the object (layers must not
    /// swallow delete).
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_memory_delete_through_operator_cache() {
        let storage = memory_storage();
        storage
            .write("memory:/del/x", Bytes::from("x"))
            .await
            .expect("write");
        assert!(storage.exists("memory:/del/x").await.expect("exists"));
        storage.delete("memory:/del/x").await.expect("delete");
        assert!(
            !storage
                .exists("memory:/del/x")
                .await
                .expect("exists after delete"),
            "delete via cached Operator must remove the object"
        );
    }

    /// Range reader via cached Operator returns the requested slice.
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_memory_reader_range_through_operator_cache() {
        let storage = memory_storage();
        storage
            .write("memory:/range/data", Bytes::from("abcdefgh"))
            .await
            .expect("write");
        let reader = storage.reader("memory:/range/data").await.expect("reader");
        let slice = reader.read(2..6).await.expect("range read");
        assert_eq!(slice.as_ref(), b"cdef");
    }

    /// Concurrent first-accesses for the same key must build once and all share
    /// the finished Operator (double-checked locking under the mutex).
    #[cfg(feature = "opendal-memory")]
    #[test]
    fn test_operator_cache_concurrent_same_key_builds_once() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let cache = OperatorCache::default();
        let builds = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            let builds = Arc::clone(&builds);
            handles.push(thread::spawn(move || {
                cache
                    .get_or_insert_with("concurrent".to_string(), || {
                        builds.fetch_add(1, Ordering::SeqCst);
                        // Slight delay so threads pile up on the miss path.
                        thread::sleep(std::time::Duration::from_millis(2));
                        memory_config_build()
                    })
                    .expect("concurrent get_or_insert_with")
            }));
        }
        let ops: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect();
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "exactly one build under concurrent first-access"
        );
        for op in &ops[1..] {
            assert!(
                Arc::ptr_eq(ops[0].inner(), op.inner()),
                "all concurrent callers must share the same finished Operator"
            );
        }
    }

    /// Layered cached memory Operator still reads back what was written (Retry +
    /// ConcurrentLimit must not alter payload correctness).
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_memory_write_read_roundtrip_through_operator_cache() {
        let storage = memory_storage();
        let payload = Bytes::from(vec![7u8; 4096]);
        storage
            .write("memory:/rt/blob.bin", payload.clone())
            .await
            .expect("write through cache");
        let read_back = storage
            .read("memory:/rt/blob.bin")
            .await
            .expect("read through cache");
        assert_eq!(read_back, payload);
        assert!(storage.exists("memory:/rt/blob.bin").await.expect("exists"));
        let meta = storage
            .metadata("memory:/rt/blob.bin")
            .await
            .expect("metadata");
        assert_eq!(meta.size, 4096);
    }

    /// S3 bucket-root relative path is empty string; scheme prefix forms list base.
    #[cfg(feature = "opendal-s3")]
    #[test]
    fn test_s3_bucket_root_relative_path_is_empty() {
        use iceberg::io::{S3_DISABLE_CONFIG_LOAD, S3_DISABLE_EC2_METADATA, S3_REGION};

        let props: std::collections::HashMap<String, String> = [
            (S3_REGION, "us-east-1"),
            (S3_DISABLE_CONFIG_LOAD, "true"),
            (S3_DISABLE_EC2_METADATA, "true"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let config = crate::s3::s3_config_parse(props).expect("offline s3 config");
        let storage = OpenDalStorage::S3 {
            configured_scheme: "s3".to_string(),
            config: std::sync::Arc::new(config),
            customized_credential_load: None,
            operator_cache: OperatorCache::default(),
        };

        let (op, rel) = storage
            .create_operator(&"s3://root-bucket/")
            .expect("bucket root must resolve");
        assert_eq!(rel, "", "relative path for bucket root must be empty");
        assert_eq!(op.info().name(), "root-bucket");

        // After warm, nested key still correct.
        let (op2, rel2) = storage
            .create_operator(&"s3://root-bucket/prefix/obj")
            .expect("nested after root warm");
        assert_eq!(rel2, "prefix/obj");
        assert!(std::sync::Arc::ptr_eq(op.inner(), op2.inner()));
    }

    /// OperatorCache invokes the build closure once per key (miss), never on hit.
    #[cfg(feature = "opendal-memory")]
    #[test]
    fn test_operator_cache_build_runs_once_per_key() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let cache = OperatorCache::default();
        let builds = AtomicUsize::new(0);

        let op1 = cache
            .get_or_insert_with("once".to_string(), || {
                builds.fetch_add(1, Ordering::SeqCst);
                memory_config_build()
            })
            .expect("first miss builds");
        let op2 = cache
            .get_or_insert_with("once".to_string(), || {
                builds.fetch_add(1, Ordering::SeqCst);
                memory_config_build()
            })
            .expect("hit must not rebuild");
        let op3 = cache
            .get_or_insert_with("once".to_string(), || {
                builds.fetch_add(1, Ordering::SeqCst);
                memory_config_build()
            })
            .expect("second hit must not rebuild");
        assert_eq!(builds.load(Ordering::SeqCst), 1, "build once per key");
        assert!(std::sync::Arc::ptr_eq(op1.inner(), op2.inner()));
        assert!(std::sync::Arc::ptr_eq(op1.inner(), op3.inner()));

        let _other = cache
            .get_or_insert_with("other".to_string(), || {
                builds.fetch_add(1, Ordering::SeqCst);
                memory_config_build()
            })
            .expect("different key builds again");
        assert_eq!(
            builds.load(Ordering::SeqCst),
            2,
            "distinct key builds once more"
        );
    }

    /// Trailing-slash list prefix must not double-slash locations (orphan path equality).
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_opendal_memory_list_trailing_slash_prefix_locations() {
        let storage = memory_storage();
        storage
            .write("memory:/dir/a.txt", Bytes::from("a"))
            .await
            .expect("write");
        let listed = storage
            .list("memory:/dir/")
            .await
            .expect("list trailing-slash prefix");
        assert_eq!(listed.len(), 1);
        assert_eq!(
            listed[0].location, "memory:/dir/a.txt",
            "trailing-slash prefix must not produce double-slash locations"
        );
        assert!(!listed[0].location.contains("//dir"), "no scheme-local //");
        assert!(!listed[0].location.contains("dir//"), "no dir//");
    }

    /// GCS offline: same-bucket paths share a cached Operator; other buckets do not.
    #[cfg(feature = "opendal-gcs")]
    #[test]
    fn test_operator_cache_reuses_operator_for_same_gcs_bucket() {
        use iceberg::io::{GCS_DISABLE_CONFIG_LOAD, GCS_DISABLE_VM_METADATA, GCS_NO_AUTH};

        let props: std::collections::HashMap<String, String> = [
            (GCS_NO_AUTH, "true"),
            (GCS_DISABLE_CONFIG_LOAD, "true"),
            (GCS_DISABLE_VM_METADATA, "true"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let config = crate::gcs::gcs_config_parse(props).expect("offline gcs config");
        let storage = OpenDalStorage::Gcs {
            config: std::sync::Arc::new(config),
            operator_cache: OperatorCache::default(),
        };

        let (op1, rel1) = storage
            .create_operator(&"gs://gcs-bucket/k1")
            .expect("first gcs path");
        let (op2, rel2) = storage
            .create_operator(&"gs://gcs-bucket/nested/k2")
            .expect("second same bucket");
        assert_eq!(rel1, "k1");
        assert_eq!(rel2, "nested/k2");
        assert!(
            std::sync::Arc::ptr_eq(op1.inner(), op2.inner()),
            "same GCS bucket must reuse the cached Operator"
        );

        let (op_other, _) = storage
            .create_operator(&"gs://other-gcs-bucket/k")
            .expect("other bucket");
        assert!(
            !std::sync::Arc::ptr_eq(op1.inner(), op_other.inner()),
            "different GCS buckets must not share Operators"
        );
    }

    /// Pins the complete-list-meta selection of `(size, created_at_millis)` so a
    /// sabotage that drops mtime or size under the complete branch fails loudly.
    #[test]
    fn test_file_meta_from_complete_list_entry_size_and_mtime() {
        use opendal::{EntryMode, Metadata};

        let known_millis: i64 = 1_700_000_000_000;
        let with_mtime = Metadata::new(EntryMode::FILE)
            .with_content_length(99)
            .with_last_modified(
                opendal::raw::Timestamp::from_millisecond(known_millis).expect("ts"),
            );
        assert!(list_entry_metadata_complete(&with_mtime));
        assert_eq!(
            file_meta_from_complete_list_entry(&with_mtime),
            (99, known_millis),
            "complete entry must surface list size and mtime millis"
        );

        let size_only = Metadata::new(EntryMode::FILE).with_content_length(7);
        assert!(list_entry_metadata_complete(&size_only));
        assert_eq!(
            file_meta_from_complete_list_entry(&size_only),
            (7, 0),
            "missing last_modified must report created_at_millis = 0"
        );
    }

    /// OperatorCache recovers from a poisoned mutex so later I/O is not denied.
    #[cfg(feature = "opendal-memory")]
    #[test]
    fn test_operator_cache_recovers_from_poisoned_mutex() {
        let cache = OperatorCache::default();
        // Poison the mutex by panicking while the guard is held.
        let cache_for_panic = cache.clone();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = cache_for_panic.inner.lock().expect("lock before poison");
            panic!("intentional poison for OperatorCache recovery pin");
        }));
        assert!(
            cache.inner.lock().is_err(),
            "mutex must be poisoned after the panicking critical section"
        );

        let op = cache
            .get_or_insert_with("poison-key".to_string(), memory_config_build)
            .expect("get_or_insert_with must recover from poison via into_inner");
        assert_eq!(op.info().scheme().to_string(), "memory");

        // Second lookup still works and reuses the inserted Operator.
        let op2 = cache
            .get_or_insert_with("poison-key".to_string(), || {
                panic!("build must not run on cache hit after poison recovery")
            })
            .expect("cache hit after poison recovery");
        assert!(
            std::sync::Arc::ptr_eq(op.inner(), op2.inner()),
            "recovered cache must still share the finished Operator"
        );
    }

    /// Serde skips `operator_cache` (`#[serde(skip, default)]`). Simulate the
    /// post-deserialize state with a fresh empty cache sharing the same config:
    /// create_operator must rebuild Operators, and clones must share that new cache
    /// without retaining the pre-skip Operator identity.
    #[cfg(feature = "opendal-s3")]
    #[test]
    fn test_operator_cache_empty_after_serde_skip_rebuilds() {
        use iceberg::io::{S3_DISABLE_CONFIG_LOAD, S3_DISABLE_EC2_METADATA, S3_REGION};

        let props: std::collections::HashMap<String, String> = [
            (S3_REGION, "us-east-1"),
            (S3_DISABLE_CONFIG_LOAD, "true"),
            (S3_DISABLE_EC2_METADATA, "true"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let config = crate::s3::s3_config_parse(props).expect("offline s3 config");
        let config = std::sync::Arc::new(config);
        let storage = OpenDalStorage::S3 {
            configured_scheme: "s3".to_string(),
            config: config.clone(),
            customized_credential_load: None,
            operator_cache: OperatorCache::default(),
        };

        let (warm_op, warm_rel) = storage
            .create_operator(&"s3://serde-bucket/pre")
            .expect("warm create_operator");
        assert_eq!(warm_rel, "pre");
        assert_eq!(warm_op.info().name(), "serde-bucket");

        // Post-serde-skip shape: same durable config, empty operator_cache.
        let restored = OpenDalStorage::S3 {
            configured_scheme: "s3".to_string(),
            config,
            customized_credential_load: None,
            operator_cache: OperatorCache::default(),
        };

        let (op1, rel1) = restored
            .create_operator(&"s3://serde-bucket/post/nested")
            .expect("post-skip create_operator must rebuild operator from config");
        assert_eq!(rel1, "post/nested");
        assert_eq!(op1.info().name(), "serde-bucket");

        let cloned = restored.clone();
        let (op2, rel2) = cloned
            .create_operator(&"s3a://serde-bucket/other")
            .expect("clone after rebuild must share the fresh cache");
        assert_eq!(rel2, "other");
        assert!(
            std::sync::Arc::ptr_eq(op1.inner(), op2.inner()),
            "fresh cache after serde-skip must be shared across clones"
        );
        assert!(
            !std::sync::Arc::ptr_eq(warm_op.inner(), op1.inner()),
            "empty cache must not retain the pre-skip Operator identity"
        );
    }

    /// Nested keys keep correct relative paths after the bucket Operator is cached,
    /// including across S3 scheme aliases (URL strip, not operator-root strip).
    #[cfg(feature = "opendal-s3")]
    #[test]
    fn test_operator_cache_nested_relative_path_after_warm() {
        use iceberg::io::{S3_DISABLE_CONFIG_LOAD, S3_DISABLE_EC2_METADATA, S3_REGION};

        let props: std::collections::HashMap<String, String> = [
            (S3_REGION, "us-east-1"),
            (S3_DISABLE_CONFIG_LOAD, "true"),
            (S3_DISABLE_EC2_METADATA, "true"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let config = crate::s3::s3_config_parse(props).expect("offline s3 config");
        let storage = OpenDalStorage::S3 {
            configured_scheme: "s3".to_string(),
            config: std::sync::Arc::new(config),
            customized_credential_load: None,
            operator_cache: OperatorCache::default(),
        };

        let (op_warm, rel_warm) = storage
            .create_operator(&"s3://nest-bucket/seed")
            .expect("warm");
        assert_eq!(rel_warm, "seed");

        let (op_nested, rel_nested) = storage
            .create_operator(&"s3://nest-bucket/a/b/c.parquet")
            .expect("nested after warm");
        assert_eq!(rel_nested, "a/b/c.parquet");
        assert!(
            std::sync::Arc::ptr_eq(op_warm.inner(), op_nested.inner()),
            "nested path must reuse the warm bucket Operator"
        );

        let (op_alias, rel_alias) = storage
            .create_operator(&"s3a://nest-bucket/x/y/z")
            .expect("alias nested after warm");
        assert_eq!(rel_alias, "x/y/z");
        assert!(std::sync::Arc::ptr_eq(op_warm.inner(), op_alias.inner()));
    }

    /// Local FS list goes through the incomplete-meta → stat path (OpenDAL FS list
    /// entries carry mode only). Pins empty + non-empty sizes under Wave C list.
    #[cfg(feature = "opendal-fs")]
    #[tokio::test]
    async fn test_opendal_fs_list_sizes_via_stat_fallback() {
        let root = std::env::temp_dir().join(format!(
            "iceberg-opendal-fs-list-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("sub")).expect("mkdir");
        std::fs::write(root.join("empty.txt"), b"").expect("write empty");
        std::fs::write(root.join("payload.txt"), b"hello").expect("write payload");
        std::fs::write(root.join("sub").join("nested.txt"), b"ab").expect("write nested");

        let storage = OpenDalStorage::LocalFs {
            operator_cache: OperatorCache::default(),
        };
        // file:/ + absolute path (strip "file:/" keeps leading / on Unix).
        let prefix = format!("file:{}", root.display());
        let listed = storage.list(&prefix).await;
        let _ = std::fs::remove_dir_all(&root);
        let mut listed = listed.expect("FS list must succeed via stat fallback");
        listed.sort_by(|a, b| a.location.cmp(&b.location));

        let by_name = |name: &str| {
            listed
                .iter()
                .find(|f| f.location.ends_with(name))
                .unwrap_or_else(|| panic!("missing {name} in {listed:?}"))
        };
        assert_eq!(by_name("empty.txt").size, 0, "empty file size via stat");
        assert_eq!(by_name("payload.txt").size, 5, "payload size via stat");
        assert_eq!(by_name("nested.txt").size, 2, "nested size via stat");
        assert_eq!(listed.len(), 3, "exactly the three files, no dirs");
    }

    /// Memory OpenDAL list entries only carry mode (no size / mtime). List must
    /// still return correct sizes via the `stat` fallback — pins the incomplete path.
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_opendal_memory_list_uses_stat_fallback_for_incomplete_list_meta() {
        let storage = memory_storage();
        storage
            .write("memory:/meta/empty.txt", Bytes::from(""))
            .await
            .expect("write empty");
        storage
            .write("memory:/meta/payload.txt", Bytes::from("hello"))
            .await
            .expect("write payload");

        let mut listed = storage
            .list("memory:/meta")
            .await
            .expect("list must succeed via stat fallback");
        listed.sort_by(|a, b| a.location.cmp(&b.location));

        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].location, "memory:/meta/empty.txt");
        assert_eq!(listed[0].size, 0, "empty file size via stat");
        assert_eq!(listed[1].location, "memory:/meta/payload.txt");
        assert_eq!(listed[1].size, 5, "payload size via stat");
    }

    /// FK4.2: parse `client.list-stat-concurrency` (default 16; 0 → 1; bad → default).
    #[test]
    fn test_fk4_2_parse_list_stat_concurrency() {
        assert_eq!(
            parse_list_stat_concurrency(&HashMap::new()),
            DEFAULT_LIST_STAT_CONCURRENCY
        );
        let mut props = HashMap::new();
        props.insert(CLIENT_LIST_STAT_CONCURRENCY.to_string(), "32".to_string());
        assert_eq!(parse_list_stat_concurrency(&props), 32);
        props.insert(CLIENT_LIST_STAT_CONCURRENCY.to_string(), "0".to_string());
        assert_eq!(
            parse_list_stat_concurrency(&props),
            1,
            "0 clamps to sequential"
        );
        props.insert(CLIENT_LIST_STAT_CONCURRENCY.to_string(), "nope".to_string());
        assert_eq!(
            parse_list_stat_concurrency(&props),
            DEFAULT_LIST_STAT_CONCURRENCY
        );
        assert_eq!(
            OperatorCache::default().list_stat_concurrency,
            DEFAULT_LIST_STAT_CONCURRENCY
        );
        assert_eq!(
            OperatorCache::default()
                .with_list_stat_concurrency(0)
                .list_stat_concurrency,
            1
        );
    }

    /// FK4.2: factory wires the knob from StorageConfig into the storage instance,
    /// and FileIOBuilder→list exercises the factory-built `Arc<dyn Storage>` path.
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_fk4_2_factory_honors_list_stat_concurrency_prop() {
        let config = StorageConfig::new().with_prop(CLIENT_LIST_STAT_CONCURRENCY, "4");
        let concrete = OpenDalStorage::Memory {
            operator: memory_config_build().expect("memory op"),
            operator_cache: operator_cache_from_config(&config),
        };
        assert_eq!(concrete.list_stat_concurrency(), 4);
        let defaulted = OpenDalStorage::Memory {
            operator: memory_config_build().expect("memory op"),
            operator_cache: operator_cache_from_config(&StorageConfig::new()),
        };
        assert_eq!(
            defaulted.list_stat_concurrency(),
            DEFAULT_LIST_STAT_CONCURRENCY
        );

        // End-to-end: factory.build props flow into list (incomplete-stat path).
        use iceberg::io::FileIOBuilder;
        let file_io = FileIOBuilder::new(std::sync::Arc::new(OpenDalStorageFactory::Memory))
            .with_prop(CLIENT_LIST_STAT_CONCURRENCY, "4")
            .build();
        file_io
            .new_output("memory:/fk42-factory/a.txt")
            .expect("output")
            .write(Bytes::from("hi"))
            .await
            .expect("write");
        let listed = file_io
            .list("memory:/fk42-factory")
            .await
            .expect("list via factory-built FileIO");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].size, 2);
    }

    /// FK4.2: concurrent incomplete stats (concurrency=1 and default) return the same
    /// ordered sizes for a multi-file memory list (all entries incomplete → all HEADs).
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_fk4_2_concurrent_list_stat_sizes_match_sequential() {
        let payloads: Vec<(&str, Bytes)> = vec![
            ("memory:/cstat/a", Bytes::from("a")),
            ("memory:/cstat/b", Bytes::from("bb")),
            ("memory:/cstat/c", Bytes::from("ccc")),
            ("memory:/cstat/sub/d", Bytes::from("dddd")),
            ("memory:/cstat/empty", Bytes::from("")),
        ];

        let sequential = OpenDalStorage::Memory {
            operator: memory_config_build().expect("op"),
            operator_cache: OperatorCache::default().with_list_stat_concurrency(1),
        };
        let concurrent = OpenDalStorage::Memory {
            operator: memory_config_build().expect("op"),
            operator_cache: OperatorCache::default()
                .with_list_stat_concurrency(DEFAULT_LIST_STAT_CONCURRENCY),
        };

        for (path, body) in &payloads {
            sequential
                .write(path, body.clone())
                .await
                .expect("seq write");
            concurrent
                .write(path, body.clone())
                .await
                .expect("conc write");
        }

        let mut seq_listed = sequential.list("memory:/cstat").await.expect("seq list");
        let mut conc_listed = concurrent.list("memory:/cstat").await.expect("conc list");
        seq_listed.sort_by(|a, b| a.location.cmp(&b.location));
        conc_listed.sort_by(|a, b| a.location.cmp(&b.location));

        assert_eq!(seq_listed.len(), payloads.len());
        assert_eq!(
            seq_listed
                .iter()
                .map(|f| (f.location.as_str(), f.size))
                .collect::<Vec<_>>(),
            conc_listed
                .iter()
                .map(|f| (f.location.as_str(), f.size))
                .collect::<Vec<_>>(),
            "concurrent stats must match sequential sizes and locations"
        );
        assert_eq!(
            seq_listed
                .iter()
                .find(|f| f.location.ends_with("/a"))
                .unwrap()
                .size,
            1
        );
        assert_eq!(
            seq_listed
                .iter()
                .find(|f| f.location.ends_with("/empty"))
                .unwrap()
                .size,
            0
        );
    }

    /// FK4.2 cheap 10k-key list: every incomplete entry stats; all sizes correct.
    ///
    /// Memory LIST never carries size, so this is an N-HEAD workload (HEAD-count = N).
    /// Concurrent window defaults to 16; pin proves no dropped/duplicated keys under load.
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_fk4_2_list_10k_keys_incomplete_stat_all_sizes() {
        const N: usize = 10_000;
        let storage = OpenDalStorage::Memory {
            operator: memory_config_build().expect("op"),
            operator_cache: OperatorCache::default()
                .with_list_stat_concurrency(DEFAULT_LIST_STAT_CONCURRENCY),
        };
        for i in 0..N {
            // Vary sizes 0..15 so a wrong/skipped stat is visible.
            let size = i % 16;
            let body = Bytes::from(vec![b'x'; size]);
            storage
                .write(&format!("memory:/bulk10k/{i:05}"), body)
                .await
                .unwrap_or_else(|e| panic!("write {i}: {e}"));
        }

        let listed = storage
            .list("memory:/bulk10k")
            .await
            .expect("10k list via concurrent incomplete stat");
        assert_eq!(
            listed.len(),
            N,
            "must return exactly N keys (no drop/dup under concurrent stat)"
        );
        // HEAD-count disclosure for the ledger: memory incomplete ⇒ one stat per key.
        let head_count = listed.len();
        assert_eq!(
            head_count, N,
            "HEAD-count == key count on incomplete-list backend"
        );

        let mut by_name: HashMap<String, u64> = HashMap::with_capacity(N);
        for f in &listed {
            by_name.insert(f.location.clone(), f.size);
        }
        for i in 0..N {
            let loc = format!("memory:/bulk10k/{i:05}");
            let expected = (i % 16) as u64;
            assert_eq!(
                by_name.get(&loc).copied(),
                Some(expected),
                "size mismatch at {loc}"
            );
        }
    }

    /// FK4.2: buffer_to_bytes is zero-copy for contiguous / single-part Buffers.
    #[test]
    fn test_fk4_2_buffer_to_bytes_contiguous_zero_copy() {
        let payload = Bytes::from_static(b"contiguous-payload");
        let buf = opendal::Buffer::from(payload.clone());
        let out = buffer_to_bytes(buf);
        assert_eq!(out, payload);
        // Contiguous path shares the same allocation (Bytes refcount clone).
        assert_eq!(out.as_ptr(), payload.as_ptr());
    }

    /// FK4.2: missing object during incomplete stat fails the whole list (fail-closed;
    /// no hang / no partial success). Mutation: swallow stat Err → this pin RED.
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_fk4_2_list_stat_missing_object_fails_closed() {
        let storage = OpenDalStorage::Memory {
            operator: memory_config_build().expect("op"),
            operator_cache: OperatorCache::default().with_list_stat_concurrency(4),
        };
        // Write one live object, then inject a need_stat path that does not exist.
        storage
            .write("memory:/miss/live.txt", Bytes::from("ok"))
            .await
            .expect("write live");
        let (op, _) = storage
            .create_operator(&"memory:/miss/live.txt")
            .expect("op");
        let need_stat = vec![
            (0, "miss/live.txt".to_string()),
            (1, "miss/does-not-exist.txt".to_string()),
        ];
        let mut ready_meta = vec![None, None];
        let err = stat_incomplete_list_entries(&op, &need_stat, 4, &mut ready_meta)
            .await
            .expect_err("missing object must fail list-stat");
        assert!(
            format!("{err:#}").contains("Failure in doing io operation")
                || format!("{err}").contains("Failure in doing io operation"),
            "unexpected error shape: {err:#}"
        );
    }

    /// FK4.2: out-of-range slot index from a stat result hard-fails (C1-Q-002).
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_fk4_2_list_stat_oob_slot_hard_fails() {
        let storage = memory_storage();
        storage
            .write("memory:/oob/x", Bytes::from("x"))
            .await
            .expect("write");
        let (op, _) = storage.create_operator(&"memory:/oob/x").expect("op");
        // Slot 99 does not exist in ready_meta of length 1.
        let need_stat = vec![(99, "oob/x".to_string())];
        let mut ready_meta = vec![None];
        let err = stat_incomplete_list_entries(&op, &need_stat, 1, &mut ready_meta)
            .await
            .expect_err("OOB slot must hard-fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("out-of-range slot index"),
            "expected OOB slot message, got: {msg}"
        );
    }

    /// FK4.2: list_stat_concurrency is preserved across OpenDalStorage clone
    /// (copied with OperatorCache; not reset to default).
    #[cfg(feature = "opendal-memory")]
    #[test]
    fn test_fk4_2_concurrency_survives_storage_clone() {
        let storage = OpenDalStorage::Memory {
            operator: memory_config_build().expect("op"),
            operator_cache: OperatorCache::default().with_list_stat_concurrency(7),
        };
        assert_eq!(storage.list_stat_concurrency(), 7);
        let cloned = storage.clone();
        assert_eq!(cloned.list_stat_concurrency(), 7);
        assert_eq!(storage.list_stat_concurrency(), 7);
    }

    /// FK4.2: empty need_stat is a successful no-op (complete-list-meta only path).
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_fk4_2_stat_empty_need_stat_is_noop() {
        let mut ready_meta: Vec<Option<(u64, i64)>> = vec![Some((1, 0)), Some((2, 0))];
        // Operator unused when need_stat is empty — use a fresh memory op anyway.
        let op = memory_config_build().expect("memory op");
        stat_incomplete_list_entries(&op, &[], 16, &mut ready_meta)
            .await
            .expect("empty need_stat must succeed");
        assert_eq!(ready_meta, vec![Some((1, 0)), Some((2, 0))]);
    }

    /// FK4.2: concurrent stats fill only incomplete slots and must not clobber
    /// already-complete list-meta slots (interleaved ready/need_stat).
    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_fk4_2_stat_does_not_clobber_complete_slots() {
        let storage = memory_storage();
        storage
            .write("memory:/mix/a", Bytes::from("aa"))
            .await
            .expect("write a");
        storage
            .write("memory:/mix/b", Bytes::from("bbbb"))
            .await
            .expect("write b");
        let (op, _) = storage.create_operator(&"memory:/mix/a").expect("op");
        // Slots: 0 complete, 1 need stat(a), 2 complete, 3 need stat(b)
        let mut ready_meta = vec![Some((10, 100)), None, Some((30, 300)), None];
        let need_stat = vec![(1, "mix/a".to_string()), (3, "mix/b".to_string())];
        stat_incomplete_list_entries(&op, &need_stat, 8, &mut ready_meta)
            .await
            .expect("mixed slots");
        assert_eq!(
            ready_meta[0],
            Some((10, 100)),
            "complete slot 0 must be untouched"
        );
        assert_eq!(
            ready_meta[2],
            Some((30, 300)),
            "complete slot 2 must be untouched"
        );
        assert_eq!(
            ready_meta[1].expect("slot 1 filled").0,
            2,
            "stat size for mix/a"
        );
        assert_eq!(
            ready_meta[3].expect("slot 3 filled").0,
            4,
            "stat size for mix/b"
        );
    }

    /// Operator cache reuses the same finished Operator for two paths on the same
    /// memory backend, and clones of storage share the cache.
    #[cfg(feature = "opendal-memory")]
    #[test]
    fn test_operator_cache_reuses_operator_for_same_backend() {
        let storage = memory_storage();
        let (op1, rel1) = storage
            .create_operator(&"memory:/a/x")
            .expect("first create_operator");
        let (op2, rel2) = storage
            .create_operator(&"memory:/a/y")
            .expect("second create_operator");
        assert_eq!(rel1, "a/x");
        assert_eq!(rel2, "a/y");
        assert_eq!(op1.info().name(), op2.info().name());
        assert!(
            Arc::ptr_eq(op1.inner(), op2.inner()),
            "same-backend paths must share the cached Operator accessor"
        );

        let cloned = storage.clone();
        let (op3, _) = cloned
            .create_operator(&"memory:/a/z")
            .expect("clone create_operator");
        assert!(
            Arc::ptr_eq(op1.inner(), op3.inner()),
            "cloned OpenDalStorage must share the operator cache"
        );
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

        use crate::{OpenDalStorage, OpenDalStorageFactory, OperatorCache};

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
                operator_cache: OperatorCache::default(),
            }
        }

        /// Operator cache: two keys in the same bucket share one Operator; a
        /// different bucket gets a different Operator. Offline (no AWS I/O).
        #[test]
        fn test_operator_cache_reuses_operator_for_same_bucket() {
            let storage = s3_storage("s3");
            let (op1, rel1) = storage
                .create_operator(&"s3://my-bucket/k1")
                .expect("first path");
            let (op2, rel2) = storage
                .create_operator(&"s3://my-bucket/k2")
                .expect("second path same bucket");
            assert_eq!(rel1, "k1");
            assert_eq!(rel2, "k2");
            assert_eq!(op1.info().name(), "my-bucket");
            assert_eq!(op2.info().name(), "my-bucket");
            assert!(
                Arc::ptr_eq(op1.inner(), op2.inner()),
                "same bucket must reuse the cached Operator"
            );

            let (op_other, _) = storage
                .create_operator(&"s3://other-bucket/k")
                .expect("other bucket");
            assert_eq!(op_other.info().name(), "other-bucket");
            assert!(
                !Arc::ptr_eq(op1.inner(), op_other.inner()),
                "different buckets must not share Operators"
            );

            // Clone shares the cache.
            let cloned = storage.clone();
            let (op3, _) = cloned
                .create_operator(&"s3a://my-bucket/k3")
                .expect("clone + s3a alias");
            assert!(
                Arc::ptr_eq(op1.inner(), op3.inner()),
                "cloned storage must share the operator cache for the same bucket"
            );
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
