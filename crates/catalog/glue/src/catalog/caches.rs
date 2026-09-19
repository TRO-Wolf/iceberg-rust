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
use iceberg::{
    CacheScope, CommitBaseLoadPlan, Error, ErrorKind, NamespaceIdent, Result, TableCommit,
    TableIdent, TableMetadataCache, commit_base_conflict_error, load_or_fetch_table_metadata,
    plan_commit_base_load,
};

#[cfg(test)]
use super::PointerSource;
use super::{GlueCatalog, GlueCatalogBuilder};
use crate::error::from_aws_sdk_error;
use crate::utils::{get_metadata_location, validate_namespace};
use crate::with_catalog_id;

#[allow(missing_docs)]
impl GlueCatalogBuilder {
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

impl GlueCatalog {
    pub(super) fn with_cache_options(
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

    pub(super) fn table_builder(&self) -> TableBuilder {
        let builder = Table::builder().file_io(self.file_io());
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

    pub(super) async fn cache_put(
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

    #[cfg(test)]
    pub(super) fn with_pointer_source(mut self, pointer_source: PointerSource) -> Self {
        self.pointer_source = Some(pointer_source);
        self
    }

    #[cfg(test)]
    pub(super) fn with_drop_source(mut self, source: super::DropSource) -> Self {
        self.drop_source = Some(source);
        self
    }

    #[cfg(test)]
    pub(super) fn with_create_source(mut self, source: super::CreateSource) -> Self {
        self.create_source = Some(source);
        self
    }

    #[cfg(test)]
    pub(super) fn with_file_io_for_tests(mut self, file_io: FileIO) -> Self {
        self.file_io = file_io;
        self
    }

    pub(super) async fn get_table_pointer(
        &self,
        table: &TableIdent,
    ) -> Result<(String, Option<String>)> {
        let db_name = validate_namespace(table.namespace())?;
        let table_name = table.name();

        #[cfg(test)]
        if let Some(pointer_source) = &self.pointer_source {
            return pointer_source(table);
        }
        #[cfg(test)]
        if let Some(harness) = &self.outcome_harness {
            return Ok(harness.pointer());
        }

        let builder = self
            .client
            .0
            .get_table()
            .database_name(&db_name)
            .name(table_name);
        let builder = with_catalog_id!(builder, self.config);

        let glue_table_output = builder.send().await.map_err(from_aws_sdk_error)?;

        let glue_table = glue_table_output.table().ok_or_else(|| {
            Error::new(
                ErrorKind::TableNotFound,
                format!(
                    "Table object for database: {db_name} and table: {table_name} does not exist"
                ),
            )
        })?;

        let version_id = glue_table.version_id.clone();
        let metadata_location = get_metadata_location(&glue_table.parameters)?;
        Ok((metadata_location, version_id))
    }

    pub(super) async fn load_table_with_version_id(
        &self,
        table: &TableIdent,
    ) -> Result<(Table, Option<String>)> {
        let db_name = validate_namespace(table.namespace())?;
        let table_name = table.name();
        #[cfg(test)]
        if let Some(harness) = &self.outcome_harness {
            let loaded = harness.table();
            let version_id = harness.pointer().1;
            let rebound = self
                .table_builder()
                .metadata_location(
                    loaded
                        .metadata_location()
                        .ok_or_else(|| {
                            Error::new(
                                ErrorKind::Unexpected,
                                format!(
                                    "Table object for database: {db_name} and table: {table_name} is missing a metadata location"
                                ),
                            )
                        })?
                        .to_string(),
                )
                .metadata(loaded.metadata_ref())
                .identifier(TableIdent::new(
                    NamespaceIdent::new(db_name),
                    table_name.to_owned(),
                ))
                .build()?;
            return Ok((rebound, version_id));
        }
        let (metadata_location, version_id) = self.get_table_pointer(table).await?;

        let metadata = load_or_fetch_table_metadata(
            &self.file_io,
            &self.cache_scope,
            &metadata_location,
            self.table_metadata_cache.as_deref(),
            version_id.as_deref(),
        )
        .await?;

        let table = self
            .table_builder()
            .metadata_location(metadata_location)
            .metadata(metadata)
            .identifier(TableIdent::new(
                NamespaceIdent::new(db_name),
                table_name.to_owned(),
            ))
            .build()?;

        Ok((table, version_id))
    }

    pub(super) async fn resolve_commit_base(
        &self,
        table_ident: &TableIdent,
        commit: &mut TableCommit,
    ) -> Result<(Table, Option<String>, String)> {
        let (service_location, version_id) = self.get_table_pointer(table_ident).await?;
        let base_loc = commit.base_metadata_location().map(str::to_string);
        let provided = commit.take_base_table();
        let provided_loc = provided
            .as_ref()
            .and_then(|t| t.metadata_location().map(str::to_string));

        match plan_commit_base_load(
            &service_location,
            base_loc.as_deref(),
            provided_loc.as_deref(),
        ) {
            CommitBaseLoadPlan::ReuseProvided => {
                let provided = provided.ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "commit base-load plan is ReuseProvided but no base table was supplied",
                    )
                })?;
                let db_name = validate_namespace(table_ident.namespace())?;
                let table = self
                    .table_builder()
                    .metadata_location(service_location.clone())
                    .metadata(provided.metadata_ref())
                    .identifier(TableIdent::new(
                        NamespaceIdent::new(db_name),
                        table_ident.name().to_owned(),
                    ))
                    .build()?;
                Ok((table, version_id, service_location))
            }
            CommitBaseLoadPlan::Conflict => Err(commit_base_conflict_error(
                table_ident,
                base_loc.as_deref(),
                &service_location,
            )),
            CommitBaseLoadPlan::FullLoad => {
                let metadata = load_or_fetch_table_metadata(
                    &self.file_io,
                    &self.cache_scope,
                    &service_location,
                    self.table_metadata_cache.as_deref(),
                    version_id.as_deref(),
                )
                .await?;
                let db_name = validate_namespace(table_ident.namespace())?;
                let table = self
                    .table_builder()
                    .metadata_location(service_location.clone())
                    .metadata(metadata)
                    .identifier(TableIdent::new(
                        NamespaceIdent::new(db_name),
                        table_ident.name().to_owned(),
                    ))
                    .build()?;
                Ok((table, version_id, service_location))
            }
        }
    }
}

fn build_object_cache(file_io: &FileIO, bytes: Option<u64>) -> Option<Arc<ObjectCache>> {
    bytes
        .filter(|bytes| *bytes > 0)
        .map(|bytes| Arc::new(ObjectCache::new_with_capacity(file_io.clone(), bytes)))
}
