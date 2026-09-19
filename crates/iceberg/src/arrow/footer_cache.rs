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

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use moka::ops::compute;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};

use crate::arrow::open_parquet::{OpenParquetError, page_index_policy};
use crate::arrow::reader::{ArrowFileReader, ParquetReadOptions};
use crate::catalog::CacheScope;
use crate::{Error, ErrorKind};

const DEFAULT_MAX_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FooterKey {
    scope: CacheScope,
    path: Arc<str>,
    file_size_in_bytes: u64,
}

#[derive(Debug, Clone)]
struct CachedFooter {
    metadata: Arc<ParquetMetaData>,
    index_checked: bool,
}

fn footer_weight(_key: &FooterKey, entry: &CachedFooter) -> u32 {
    u32::try_from(entry.metadata.memory_size()).unwrap_or(u32::MAX)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(missing_docs)]
pub struct ParquetFooterCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub fetches: u64,
    pub upgrades: u64,
    pub evictions: u64,
}

#[derive(Debug)]
#[allow(missing_docs)]
pub struct ParquetFooterCache {
    entries: moka::future::Cache<FooterKey, CachedFooter>,
    hits: AtomicU64,
    misses: AtomicU64,
    fetches: AtomicU64,
    upgrades: AtomicU64,
    evictions: Arc<AtomicU64>,
}

impl Default for ParquetFooterCache {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl ParquetFooterCache {
    pub fn new() -> Self {
        Self::with_max_bytes(DEFAULT_MAX_BYTES)
    }

    pub fn with_max_bytes(max_bytes: u64) -> Self {
        let evictions = Arc::new(AtomicU64::new(0));
        let counted = evictions.clone();
        Self {
            entries: moka::future::Cache::builder()
                .weigher(footer_weight)
                .max_capacity(max_bytes)
                .eviction_listener(move |_key, _value, cause| {
                    if cause.was_evicted() {
                        counted.fetch_add(1, Ordering::Relaxed);
                    }
                })
                .build(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            fetches: AtomicU64::new(0),
            upgrades: AtomicU64::new(0),
            evictions,
        }
    }

    pub fn stats(&self) -> ParquetFooterCacheStats {
        ParquetFooterCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            fetches: self.fetches.load(Ordering::Relaxed),
            upgrades: self.upgrades.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    #[cfg(test)]
    pub(crate) async fn run_pending_tasks(&self) {
        self.entries.run_pending_tasks().await;
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> u64 {
        self.entries.entry_count()
    }

    #[cfg(test)]
    pub(crate) fn weighted_size(&self) -> u64 {
        self.entries.weighted_size()
    }
}

#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct TableFooterCache {
    cache: Arc<ParquetFooterCache>,
    scope: CacheScope,
}

#[allow(missing_docs)]
impl TableFooterCache {
    pub fn new(cache: Arc<ParquetFooterCache>, scope: CacheScope) -> Self {
        Self { cache, scope }
    }

    pub fn shared(&self) -> Arc<ParquetFooterCache> {
        self.cache.clone()
    }

    pub(crate) async fn seed(
        &self,
        path: &str,
        file_size_in_bytes: u64,
        metadata: Arc<ParquetMetaData>,
    ) {
        let key = self.key(path, file_size_in_bytes);
        let index_checked = metadata.column_index().is_some() && metadata.offset_index().is_some();
        self.cache
            .entries
            .entry_by_ref(&key)
            .or_insert(CachedFooter {
                metadata,
                index_checked,
            })
            .await;
    }

    fn key(&self, path: &str, file_size_in_bytes: u64) -> FooterKey {
        FooterKey {
            scope: self.scope.clone(),
            path: Arc::from(path),
            file_size_in_bytes,
        }
    }

    pub(crate) async fn footer_or_fetch(
        &self,
        path: &str,
        file_size_in_bytes: u64,
        options: ParquetReadOptions,
        reader: &mut ArrowFileReader,
    ) -> std::result::Result<Arc<ParquetMetaData>, OpenParquetError> {
        let need_index = options.preload_page_index();
        let key = self.key(path, file_size_in_bytes);
        let entry = match self.cache.entries.get(&key).await {
            Some(entry) => {
                self.cache.hits.fetch_add(1, Ordering::Relaxed);
                entry
            }
            None => {
                self.cache.misses.fetch_add(1, Ordering::Relaxed);
                self.cache
                    .entries
                    .try_get_with(key.clone(), async {
                        self.cache.fetches.fetch_add(1, Ordering::Relaxed);
                        let metadata = reader.get_metadata(None).await.map_err(|e| {
                            OpenParquetError::Footer(
                                Error::new(
                                    ErrorKind::Unexpected,
                                    "Failed to load Parquet metadata",
                                )
                                .with_source(e),
                            )
                        })?;
                        Ok::<CachedFooter, OpenParquetError>(CachedFooter {
                            metadata,
                            index_checked: need_index,
                        })
                    })
                    .await
                    .map_err(|e| match e.as_ref() {
                        OpenParquetError::Footer(e) => OpenParquetError::Footer(shared_error(e)),
                        OpenParquetError::Other(e) => OpenParquetError::Other(shared_error(e)),
                    })?
            }
        };
        if !need_index || entry.index_checked {
            return Ok(entry.metadata);
        }
        let upgraded = self
            .cache
            .entries
            .entry_by_ref(&key)
            .and_try_compute_with(|existing| async move {
                if existing.as_ref().is_some_and(|e| e.value().index_checked) {
                    return Ok::<compute::Op<CachedFooter>, OpenParquetError>(compute::Op::Nop);
                }
                let metadata = match existing {
                    Some(e) => {
                        let mut indexed = ParquetMetaDataReader::new_with_metadata(
                            ParquetMetaData::clone(e.value().metadata.as_ref()),
                        )
                        .with_page_index_policy(page_index_policy(options.preload_page_index()))
                        .with_column_index_policy(page_index_policy(options.preload_column_index()))
                        .with_offset_index_policy(page_index_policy(
                            options.preload_offset_index(),
                        ));
                        indexed.load_page_index(reader).await.map_err(|e| {
                            OpenParquetError::Other(
                                Error::new(
                                    ErrorKind::Unexpected,
                                    "Failed to load Parquet page index",
                                )
                                .with_source(e),
                            )
                        })?;
                        let metadata = indexed.finish().map_err(|e| {
                            OpenParquetError::Other(
                                Error::new(
                                    ErrorKind::Unexpected,
                                    "Failed to load Parquet metadata",
                                )
                                .with_source(e),
                            )
                        })?;
                        self.cache.upgrades.fetch_add(1, Ordering::Relaxed);
                        Arc::new(metadata)
                    }
                    None => {
                        self.cache.fetches.fetch_add(1, Ordering::Relaxed);
                        reader.get_metadata(None).await.map_err(|e| {
                            OpenParquetError::Footer(
                                Error::new(
                                    ErrorKind::Unexpected,
                                    "Failed to load Parquet metadata",
                                )
                                .with_source(e),
                            )
                        })?
                    }
                };
                Ok(compute::Op::Put(CachedFooter {
                    metadata,
                    index_checked: true,
                }))
            })
            .await?;
        match upgraded.into_entry() {
            Some(entry) => Ok(entry.into_value().metadata),
            None => Err(OpenParquetError::Other(Error::new(
                ErrorKind::Unexpected,
                "Footer cache upgrade produced no entry",
            ))),
        }
    }
}

fn shared_error(error: &Error) -> Error {
    Error::new(error.kind(), error.message()).with_retryable(error.retryable())
}
