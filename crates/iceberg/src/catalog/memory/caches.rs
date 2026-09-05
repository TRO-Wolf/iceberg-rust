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

use super::catalog::{MemoryCatalog, MemoryCatalogBuilder};
use crate::io::FileIO;
use crate::io::object_cache::ObjectCache;
use crate::spec::TableMetadata;
use crate::table::{Table, TableBuilder};

impl MemoryCatalogBuilder {
    /// Share ONE manifest [`ObjectCache`] of `bytes` across every table this catalog loads.
    pub fn with_shared_object_cache_bytes(mut self, bytes: u64) -> Self {
        self.shared_object_cache_bytes = Some(bytes);
        self
    }
}

impl MemoryCatalog {
    pub(crate) fn shared_cache(file_io: &FileIO, bytes: Option<u64>) -> Option<Arc<ObjectCache>> {
        bytes
            .filter(|bytes| *bytes > 0)
            .map(|bytes| Arc::new(ObjectCache::new_with_capacity(file_io.clone(), bytes)))
    }

    pub(crate) fn table_builder(&self) -> TableBuilder {
        let builder = Table::builder().file_io(self.file_io.clone());
        match self.shared_object_cache.as_ref() {
            Some(cache) => builder.object_cache(cache.clone()),
            None => builder,
        }
    }

    /// Publish parsed metadata into the optional session cache (no-op when cache is OFF).
    pub(crate) fn cache_put(&self, metadata_location: &str, metadata: &TableMetadata) {
        if let Some(cache) = self.table_metadata_cache.as_ref() {
            cache.put(
                metadata_location.to_string(),
                Arc::new(metadata.clone()),
                None,
            );
        }
    }
}
