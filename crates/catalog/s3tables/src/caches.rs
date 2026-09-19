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

use iceberg::arrow::{ParquetFooterCache, TableFooterCache};
use iceberg::io::FileIO;
use iceberg::io::object_cache::ObjectCache;
use iceberg::spec::TableMetadataRef;
use iceberg::table::{Table, TableBuilder};
use iceberg::{CacheScope, TableMetadataCache};

use crate::catalog::{S3TablesCatalog, S3TablesCatalogBuilder};

#[allow(missing_docs)]
impl S3TablesCatalogBuilder {
    pub fn with_table_metadata_cache(mut self, cache: Arc<TableMetadataCache>) -> Self {
        self.table_metadata_cache = Some(cache);
        self
    }

    pub fn with_shared_object_cache_bytes(mut self, bytes: u64) -> Self {
        self.shared_object_cache_bytes = Some(bytes);
        self
    }

    pub fn with_cache_credential_context(mut self, context: String) -> Self {
        self.cache_credential_context = Some(context);
        self
    }

    pub fn with_shared_footer_cache(mut self, cache: Arc<ParquetFooterCache>) -> Self {
        self.shared_footer_cache = Some(cache);
        self
    }
}

impl S3TablesCatalog {
    pub(crate) fn with_cache_options(
        mut self,
        table_metadata_cache: Option<Arc<TableMetadataCache>>,
        shared_object_cache_bytes: Option<u64>,
        cache_credential_context: Option<String>,
        shared_footer_cache: Option<Arc<ParquetFooterCache>>,
    ) -> Self {
        self.table_metadata_cache = table_metadata_cache;
        self.shared_object_cache = build_object_cache(&self.file_io, shared_object_cache_bytes);
        self.shared_footer_cache = shared_footer_cache;
        if let Some(context) = cache_credential_context {
            self.cache_scope =
                CacheScope::new(self.cache_scope.catalog_identity().to_string(), context);
        }
        self
    }

    pub(crate) fn table_builder(&self) -> TableBuilder {
        let builder = Table::builder().file_io(self.file_io.clone());
        let builder = match self.shared_object_cache.as_ref() {
            Some(cache) => builder.object_cache(cache.clone()),
            None => builder,
        };
        match self.shared_footer_cache.as_ref() {
            Some(cache) => builder.footer_cache(TableFooterCache::new(
                cache.clone(),
                self.cache_scope.clone(),
            )),
            None => builder,
        }
    }

    pub(crate) async fn cache_put(
        &self,
        metadata_location: &str,
        metadata: TableMetadataRef,
        object_version: Option<String>,
    ) {
        if let Some(cache) = self.table_metadata_cache.as_ref() {
            cache
                .put(
                    &self.cache_scope,
                    metadata_location,
                    metadata,
                    object_version,
                    None,
                )
                .await;
        }
    }
}

pub(crate) fn build_object_cache(file_io: &FileIO, bytes: Option<u64>) -> Option<Arc<ObjectCache>> {
    bytes
        .filter(|bytes| *bytes > 0)
        .map(|bytes| Arc::new(ObjectCache::new_with_capacity(file_io.clone(), bytes)))
}
