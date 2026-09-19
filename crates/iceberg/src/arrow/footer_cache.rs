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
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};

use crate::arrow::open_parquet::{OpenParquetError, page_index_policy};
use crate::arrow::reader::{ArrowFileReader, ParquetReadOptions};
use crate::catalog::CacheScope;
use crate::{Error, ErrorKind};

const DEFAULT_MAX_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FooterKey {
    scope: CacheScope,
    path: Arc<str>,
    file_size_in_bytes: u64,
}

#[derive(Debug, Clone)]
struct CachedFooter {
    arrow_metadata: Arc<ArrowReaderMetadata>,
    index_checked: bool,
    index_attempted: bool,
}

fn has_index(metadata: &ParquetMetaData) -> bool {
    metadata.column_index().is_some() && metadata.offset_index().is_some()
}

fn base_arrow_metadata(
    metadata: Arc<ParquetMetaData>,
) -> std::result::Result<Arc<ArrowReaderMetadata>, OpenParquetError> {
    ArrowReaderMetadata::try_new(metadata, Default::default())
        .map(Arc::new)
        .map_err(|e| {
            OpenParquetError::Other(
                Error::new(ErrorKind::Unexpected, "Failed to load Parquet metadata").with_source(e),
            )
        })
}

fn footer_weight(_key: &FooterKey, entry: &CachedFooter) -> u32 {
    u32::try_from(entry.arrow_metadata.metadata().memory_size()).unwrap_or(u32::MAX)
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

    #[cfg(test)]
    pub(crate) async fn probe_index_state(
        &self,
        scope: &CacheScope,
        path: &Arc<str>,
        file_size_in_bytes: u64,
    ) -> Option<(bool, bool)> {
        self.entries
            .get(&FooterKey {
                scope: scope.clone(),
                path: Arc::clone(path),
                file_size_in_bytes,
            })
            .await
            .map(|entry| (entry.index_checked, entry.index_attempted))
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
        path: &Arc<str>,
        file_size_in_bytes: u64,
        metadata: Arc<ParquetMetaData>,
    ) {
        let index_checked = has_index(&metadata);
        let Ok(arrow_metadata) = base_arrow_metadata(metadata) else {
            return;
        };
        self.cache
            .entries
            .entry_by_ref(&self.key(path, file_size_in_bytes))
            .or_insert(CachedFooter {
                arrow_metadata,
                index_checked,
                index_attempted: false,
            })
            .await;
    }

    fn key(&self, path: &Arc<str>, file_size_in_bytes: u64) -> FooterKey {
        FooterKey {
            scope: self.scope.clone(),
            path: Arc::clone(path),
            file_size_in_bytes,
        }
    }

    pub(crate) async fn footer_or_fetch(
        &self,
        path: &Arc<str>,
        file_size_in_bytes: u64,
        options: ParquetReadOptions,
        reader: &mut ArrowFileReader,
    ) -> std::result::Result<Arc<ArrowReaderMetadata>, OpenParquetError> {
        let need_index = options.preload_page_index();
        let key = self.key(path, file_size_in_bytes);
        if let Some(entry) = self.cache.entries.get(&key).await {
            self.cache.hits.fetch_add(1, Ordering::Relaxed);
            if !need_index || entry.index_checked || entry.index_attempted {
                return Ok(Arc::clone(&entry.arrow_metadata));
            }
        } else {
            self.cache.misses.fetch_add(1, Ordering::Relaxed);
        }
        let computed = self
            .cache
            .entries
            .entry_by_ref(&key)
            .and_try_compute_with(|existing| async move {
                match existing {
                    Some(entry)
                        if !need_index
                            || entry.value().index_checked
                            || entry.value().index_attempted =>
                    {
                        Ok::<compute::Op<CachedFooter>, OpenParquetError>(compute::Op::Nop)
                    }
                    Some(entry) => {
                        let mut indexed = ParquetMetaDataReader::new_with_metadata(
                            ParquetMetaData::clone(
                                entry.value().arrow_metadata.metadata().as_ref(),
                            ),
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
                        let index_checked = has_index(&metadata);
                        let arrow_metadata = base_arrow_metadata(Arc::new(metadata))?;
                        if index_checked {
                            self.cache.upgrades.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(compute::Op::Put(CachedFooter {
                            arrow_metadata,
                            index_checked,
                            index_attempted: true,
                        }))
                    }
                    None => {
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
                        let index_checked = has_index(&metadata);
                        let arrow_metadata = base_arrow_metadata(metadata)?;
                        Ok(compute::Op::Put(CachedFooter {
                            arrow_metadata,
                            index_checked,
                            index_attempted: need_index,
                        }))
                    }
                }
            })
            .await?;
        match computed.into_entry() {
            Some(entry) => Ok(Arc::clone(&entry.into_value().arrow_metadata)),
            None => Err(OpenParquetError::Other(Error::new(
                ErrorKind::Unexpected,
                "Footer cache compute produced no entry",
            ))),
        }
    }
}
