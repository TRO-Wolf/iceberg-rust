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

use futures::lock::Mutex;

use super::catalog::MemoryCatalog;
use super::metadata_naming::MetadataNaming;
use super::namespace_state::NamespaceState;
use crate::io::FileIO;
use crate::table::Table;
use crate::{Catalog, Error, ErrorKind, Result};

pub(crate) async fn publish_create(
    catalog: &MemoryCatalog,
    naming: MetadataNaming,
    tables: &Mutex<NamespaceState>,
    table: Table,
) -> Result<Table> {
    match naming {
        MetadataNaming::Uuid => {
            let location = table.metadata_location_result()?.to_string();
            catalog.register_table(table.identifier(), location).await
        }
        MetadataNaming::Hadoop => publish_create_hadoop(catalog, tables, table).await,
    }
}

async fn publish_create_hadoop(
    catalog: &MemoryCatalog,
    tables: &Mutex<NamespaceState>,
    table: Table,
) -> Result<Table> {
    let staged_location = table.metadata_location_result()?.to_string();
    let metadata_location = MetadataNaming::Hadoop
        .first_location(table.metadata())?
        .to_string();
    let mut state = tables.lock().await;
    let slot = state.vacant_table_slot(table.identifier())?;
    table
        .metadata()
        .write_commit_metadata(&catalog.file_io, &metadata_location)
        .await?;
    slot.insert(metadata_location.clone());
    MetadataNaming::Hadoop
        .advance_version_hint(&catalog.file_io, &metadata_location)
        .await;
    drop(state);
    if let Err(error) = catalog.file_io.delete(&staged_location).await {
        tracing::warn!(
            ?error,
            staged_location,
            metadata_location,
            "published staged Hadoop metadata but failed to delete the staged file"
        );
    }
    let metadata = table.metadata_ref();
    catalog
        .cache_put(&metadata_location, metadata.clone(), None)
        .await;
    catalog
        .table_builder()
        .metadata_location(metadata_location)
        .metadata(metadata)
        .identifier(table.identifier().clone())
        .build()
}

pub(crate) async fn publish_replace(
    naming: MetadataNaming,
    file_io: &FileIO,
    tables: &Mutex<NamespaceState>,
    table: Table,
    expected_base_metadata_location: Option<String>,
) -> Result<Table> {
    let mut state = tables.lock().await;
    let ident = table.identifier().clone();
    let stored = state.get_existing_table_location(&ident)?.clone();
    if let Some(expected) = expected_base_metadata_location.as_deref()
        && stored != expected
    {
        return Err(Error::new(
            ErrorKind::CatalogCommitConflicts,
            format!(
                "Cannot publish replace for table {ident}: concurrent modification \
                 (expected base metadata location {expected}, found {stored})"
            ),
        )
        .with_retryable(true));
    }
    let updated = state.commit_table_update(table)?;
    naming
        .advance_version_hint(file_io, updated.metadata_location_result()?)
        .await;
    Ok(updated)
}
