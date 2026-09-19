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

use iceberg::spec::TableMetadata;
use iceberg::table::Table;
use iceberg::{Error, ErrorKind, Result};

use super::GlueCatalog;
#[cfg(test)]
use crate::commit_transport::glue_commit_send_landed;
use crate::commit_transport::{GlueUpdateTableCall, map_glue_commit_send};
use crate::utils::{convert_to_glue_table, validate_namespace};

pub(super) async fn publish(
    catalog: &GlueCatalog,
    table: Table,
    expected_base_metadata_location: Option<String>,
) -> Result<Table> {
    let table_ident = table.identifier().clone();
    let database_name = validate_namespace(table_ident.namespace())?;
    let (stored, version_id) = catalog.get_table_pointer(&table_ident).await?;

    if let Some(expected) = expected_base_metadata_location.as_deref()
        && stored != expected
    {
        return Err(Error::new(
            ErrorKind::CatalogCommitConflicts,
            format!(
                "Cannot publish replace for table {table_ident}: concurrent modification \
                 (expected base metadata location {expected}, found {stored})"
            ),
        )
        .with_retryable(true));
    }

    let new_metadata_location = table.metadata_location_result()?.to_string();
    let staged = TableMetadata::read_from(&catalog.file_io, &new_metadata_location)
        .await
        .map_err(|error| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot publish replace for table {table_ident}: the staged metadata \
                     file at {new_metadata_location} could not be read back"
                ),
            )
            .with_source(error)
        })?;
    if staged.uuid() != table.metadata().uuid() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Cannot publish replace for table {table_ident}: the staged metadata file \
                 at {new_metadata_location} holds table uuid {} but the table being \
                 published has uuid {}",
                staged.uuid(),
                table.metadata().uuid()
            ),
        ));
    }

    let table_input = convert_to_glue_table(
        table_ident.name(),
        new_metadata_location,
        table.metadata(),
        table.metadata().properties(),
        Some(stored),
    )?;
    let send = catalog
        .commit_transport
        .send_update_table(GlueUpdateTableCall {
            database_name,
            table_input,
            version_id,
            catalog_id: catalog.config.catalog_id.clone(),
        })
        .await;
    #[cfg(test)]
    if glue_commit_send_landed(&send)
        && let Some(harness) = &catalog.outcome_harness
    {
        harness.publish(table.clone());
    }
    map_glue_commit_send(send, &table_ident)?;

    catalog
        .cache_put(table.metadata_location_result()?, table.metadata())
        .await;

    Ok(table)
}
