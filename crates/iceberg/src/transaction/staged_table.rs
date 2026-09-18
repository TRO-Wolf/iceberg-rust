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

//! Staged create / replace table transactions (Java `Catalog.newCreateTableTransaction` /
//! `newReplaceTableTransaction` semantics).
//!
//! Writes accumulate against **uncommitted** metadata held only on FileIO. A single
//! [`StagedTableTransaction::commit`] publishes the table pointer into the catalog
//! (create) or swaps it (replace). Failure before or during publish leaves no catalog
//! create for the create path, and leaves the **original** table current for replace.
//! A published replace keeps the table's existing root location — it never relocates the
//! table, so repeated CREATE OR REPLACE cycles do not drift the location.
//!
//! A **replace** is built ON TOP OF the existing table's metadata (Java
//! `TableMetadata.buildReplacement`), not from scratch: it **retains** the table UUID, the full
//! snapshot history, and the metadata log (appended-to, never truncated), while **resetting** what
//! a replace replaces — the `main` branch ref is removed (no current snapshot) and the schema /
//! partition spec / sort order / properties / location from the `TableCreation` become the new
//! current ones. The replace-schema field-ids are taken **from the caller as provided**;
//! `last_column_id` only advances monotonically (`max` of the existing value and the caller's
//! highest field-id, never reduced). This diverges from Java's `TypeUtil.assignFreshIds`, which
//! reassigns fresh ids by **name-matching** the replacement schema against the base schema — a
//! caller supplying field-ids misaligned with the base schema's names diverges from Java (named
//! residue: a base-aware fresh-id helper is the follow-up). This is not corruption: per-snapshot
//! schema binding keeps prior history readable via each snapshot's own schema-id.
//! The format version is **preserved** across a replace unless the `TableCreation`'s properties
//! carry an explicit `format-version` directive requesting an upgrade; it is never downgraded.
//! Retaining the history keeps time-travel raw material intact while the `main` branch exposes only
//! the latest replace's data.
//!
//! Java interop battery is a disclosed non-goal of the first unit (GAP_MATRIX 🟡).

use std::str::FromStr;
use std::sync::Arc;

use crate::error::{Error, ErrorKind, Result};
use crate::expr::Predicate;
use crate::io::FileIO;
use crate::spec::{
    DataFile, FormatVersion, MAIN_BRANCH, SortOrder, TableMetadataBuilder, TableProperties,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, MetadataLocation, NamespaceIdent, TableCreation, TableIdent};

/// Publish mode for a staged table transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagedTableMode {
    /// Table must not exist in the catalog until [`StagedTableTransaction::commit`].
    Create,
    /// Table must already exist; commit swaps the metadata pointer under one catalog op.
    Replace,
}

/// Staged create or replace: data/metadata land on FileIO first; catalog publish is one step.
#[derive(Clone)]
pub struct StagedTableTransaction {
    mode: StagedTableMode,
    table: Table,
    /// For replace: the catalog metadata location observed when the transaction began (CAS base).
    base_metadata_location: Option<String>,
    pending_data_files: Vec<DataFile>,
    replace_write: bool,
}

impl StagedTableTransaction {
    /// Begin a **create** transaction. Metadata is written to FileIO but the catalog is not updated.
    ///
    /// `creation.location` must be set (absolute table root). Engines typically resolve warehouse
    /// + namespace + name before calling. `file_io` must share the catalog's storage backend.
    pub async fn begin_create(
        file_io: FileIO,
        ident: TableIdent,
        creation: TableCreation,
    ) -> Result<Self> {
        if creation.name != *ident.name() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "TableCreation.name `{}` does not match TableIdent name `{}`",
                    creation.name,
                    ident.name()
                ),
            ));
        }
        let location = creation.location.clone().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "StagedTableTransaction::begin_create requires TableCreation.location",
            )
        })?;
        let metadata = TableMetadataBuilder::from_table_creation(creation)?
            .build()?
            .metadata;
        let metadata_location = MetadataLocation::new_with_table_location(&location).to_string();
        metadata.write_to(&file_io, &metadata_location).await?;

        let table = Table::builder()
            .file_io(file_io)
            .metadata_location(metadata_location)
            .metadata(metadata)
            .identifier(ident)
            .build()?;

        Ok(Self {
            mode: StagedTableMode::Create,
            table,
            base_metadata_location: None,
            pending_data_files: Vec::new(),
            replace_write: false,
        })
    }

    /// Begin a **replace** transaction against an existing catalog table.
    ///
    /// Builds the replacement metadata ON TOP OF the existing table's metadata (Java
    /// `TableMetadata.buildReplacement`), keeping the table's **existing root location** (or the
    /// caller-provided `creation.location`). The table UUID, snapshot history, and metadata log are
    /// **retained** (the log is appended-to, never truncated); the `main` branch ref is **reset**
    /// (no current snapshot) and the `TableCreation`'s schema / partition spec / sort order /
    /// properties / location become the new current ones. The replace-schema field-ids are taken
    /// **from the caller as provided** and `last_column_id` only advances monotonically (never
    /// reduced below the base); this differs from Java's name-matching `TypeUtil.assignFreshIds`
    /// (named residue).
    ///
    /// **Format version is preserved.** `creation.format_version` is **IGNORED** on the replace
    /// path — it is indistinguishable from `TableCreation::builder()`'s V2 default, so honoring it
    /// would silently upgrade a V1 table on a default-built replace. Matching Java
    /// `buildReplacement`, the target version is derived ONLY from a `format-version` entry in
    /// `creation.properties`: absent ⇒ the existing version is kept; a higher value ⇒ upgrade; an
    /// equal value ⇒ no-op; a lower value ⇒ a hard `DataInvalid` error (never a silent downgrade);
    /// an unparsable / out-of-range value ⇒ a hard `DataInvalid` error. The `format-version` key is
    /// consumed as a directive and is NOT persisted into the table's property map (Java
    /// `persistedProperties` filters reserved properties out).
    ///
    /// A replace never relocates the table, so repeated CREATE OR REPLACE cycles leave
    /// `metadata().location()` identical every time. The original catalog entry stays current until
    /// [`StagedTableTransaction::commit`]; isolation comes from deferring the catalog pointer swap,
    /// not from a separate on-disk directory.
    pub async fn begin_replace(existing: &Table, creation: TableCreation) -> Result<Self> {
        let ident = existing.identifier().clone();
        if creation.name != *ident.name() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "TableCreation.name `{}` does not match existing table `{}`",
                    creation.name,
                    ident.name()
                ),
            ));
        }
        let base_metadata_location = existing.metadata_location_result()?.to_string();
        let existing_location = existing.metadata().location().trim_end_matches('/');
        let table_location = creation
            .location
            .clone()
            .unwrap_or_else(|| existing_location.to_string());
        let keeps_location = creation
            .location
            .as_deref()
            .is_none_or(|location| location.trim_end_matches('/') == existing_location);

        // D1: build the replacement ON TOP OF the existing metadata, mirroring Java
        // `TableMetadata.buildReplacement`:
        //   new Builder(this)                 -> new_from_metadata(previous, current_file_location):
        //                                        retains UUID + snapshot history + metadata log
        //     .upgradeFormatVersion(max)      -> max(existing, requested); never downgrades
        //     .removeRef(MAIN_BRANCH)         -> drops the main ref => no current snapshot
        //     .setCurrentSchema(fresh)        -> new schema as current, fresh IDs on the existing space
        //     .setDefaultPartitionSpec(fresh) -> new default spec, id above the existing specs
        //     .setDefaultSortOrder(fresh)     -> new default sort order, id above the existing orders
        //     .setLocation(newLocation)       -> the STABLE location resolved above (N2)
        //     .setProperties(...)
        // Passing the existing current metadata file as `current_file_location` appends it to the
        // metadata log (retained + extended, not reset).
        let previous = existing.metadata().clone();
        let previous_format_version = previous.format_version;
        let TableCreation {
            schema,
            partition_spec,
            sort_order,
            mut properties,
            ..
        } = creation;
        // D4: derive the target format version ONLY from a `format-version` PROPERTY directive,
        // mirroring Java `TableMetadata.buildReplacement` (TableMetadata.java ~730-742), which reads
        // `PropertyUtil.propertyAsInt(updatedProperties, FORMAT_VERSION, formatVersion)` (absent ⇒
        // keep the existing version) and then persists `persistedProperties(updatedProperties)` with
        // the reserved `format-version` key filtered OUT of the persisted map. `creation.format_version`
        // is intentionally IGNORED here (see the doc comment): honoring the `TableCreation::builder()`
        // V2 default would silently upgrade a V1 table on a default-built replace. Pop the key BEFORE
        // the map reaches `set_properties` (which hard-rejects reserved properties), matching Java's
        // filtering; `upgrade_format_version` then enforces Java's upgrade-only domain (no-op on
        // equal, `DataInvalid` on downgrade).
        let target_format_version =
            match properties.remove(TableProperties::PROPERTY_FORMAT_VERSION) {
                None => previous_format_version,
                Some(raw) => parse_format_version_property(&raw)?,
            };
        let partition_spec = partition_spec.unwrap_or_default();
        let sort_order = sort_order.unwrap_or_else(SortOrder::unsorted_order);

        let metadata =
            TableMetadataBuilder::new_from_metadata(previous, Some(base_metadata_location.clone()))
                .upgrade_format_version(target_format_version)?
                .remove_ref(MAIN_BRANCH)
                .add_current_schema(schema)?
                .add_default_partition_spec(partition_spec)?
                .add_sort_order(sort_order)?
                .set_default_sort_order(TableMetadataBuilder::LAST_ADDED as i64)?
                .set_location(table_location.clone())
                .set_properties(TableProperties::persisted_properties(properties))?
                .build()?
                .metadata;

        let metadata_location = match MetadataLocation::from_str(&base_metadata_location) {
            Ok(base) if keeps_location => base.with_next_version().to_string(),
            _ => MetadataLocation::new_with_table_location(&table_location).to_string(),
        };
        metadata
            .write_commit_metadata(existing.file_io(), &metadata_location)
            .await?;

        let table = Table::builder()
            .file_io(existing.file_io().clone())
            .metadata_location(metadata_location)
            .metadata(metadata)
            .identifier(ident)
            .build()?;

        Ok(Self {
            mode: StagedTableMode::Replace,
            table,
            base_metadata_location: Some(base_metadata_location),
            pending_data_files: Vec::new(),
            replace_write: false,
        })
    }

    /// Staged table handle for writers (location + schema + FileIO).
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Catalog identifier that will be published.
    pub fn identifier(&self) -> &TableIdent {
        self.table.identifier()
    }

    /// Create vs replace mode.
    pub fn mode(&self) -> StagedTableMode {
        self.mode
    }

    /// Namespace of the staged table.
    pub fn namespace(&self) -> &NamespaceIdent {
        self.table.identifier().namespace()
    }

    /// Queue data files to append on the staged metadata (no catalog publish).
    pub fn add_data_files(mut self, files: impl IntoIterator<Item = DataFile>) -> Self {
        self.pending_data_files.extend(files);
        self
    }

    /// Commit pending files with replace semantics (Java RTAS overwrite).
    pub fn with_replace_write(mut self, replace_write: bool) -> Self {
        self.replace_write = replace_write;
        self
    }

    /// Apply pending files to local metadata, write the final metadata file, then publish.
    pub async fn commit(self, catalog: &dyn Catalog) -> Result<Table> {
        let mode = self.mode;
        let base = self.base_metadata_location.clone();
        let table = self.materialize_pending().await?;
        match mode {
            StagedTableMode::Create => catalog.publish_create_table(table).await,
            StagedTableMode::Replace => catalog.publish_replace_table(table, base).await,
        }
    }

    async fn materialize_pending(self) -> Result<Table> {
        if self.pending_data_files.is_empty() && !self.replace_write {
            return Ok(self.table);
        }
        let tx = Transaction::new(&self.table);
        if self.replace_write {
            let tx = tx
                .overwrite_files()
                .overwrite_by_row_filter(Predicate::AlwaysTrue)
                .add_files(self.pending_data_files)
                .allow_empty_commit()
                .apply(tx)?;
            tx.apply_locally_in_place().await
        } else {
            let tx = tx
                .fast_append()
                .add_data_files(self.pending_data_files)
                .apply(tx)?;
            tx.apply_locally_in_place().await
        }
    }
}

impl Transaction {
    /// Apply all registered actions and write a new metadata file **without** catalog publish.
    ///
    /// Used by [`StagedTableTransaction`]: the engine finishes FileIO work first, then publishes
    /// the pointer in one catalog step.
    pub async fn apply_locally(self) -> Result<Table> {
        let current_table = self.run_actions_locally().await?;
        let next_location = MetadataLocation::from_str(current_table.metadata_location_result()?)?
            .with_next_version()
            .to_string();
        current_table
            .metadata()
            .write_commit_metadata(current_table.file_io(), &next_location)
            .await?;
        Ok(current_table.with_metadata_location(next_location))
    }

    pub(crate) async fn apply_locally_in_place(self) -> Result<Table> {
        let current_table = self.run_actions_locally().await?;
        let staged_location = current_table.metadata_location_result()?.to_string();
        current_table
            .metadata()
            .write_to(current_table.file_io(), &staged_location)
            .await?;
        Ok(current_table)
    }

    async fn run_actions_locally(self) -> Result<Table> {
        let mut current_table = self.table.clone();
        let mut existing_updates: Vec<crate::TableUpdate> = vec![];
        let mut existing_requirements: Vec<crate::TableRequirement> = vec![];

        for action in &self.actions {
            Arc::clone(action)
                .validate(self.starting_snapshot_id, &current_table)
                .await?;
        }

        for action in &self.actions {
            let action_commit = Arc::clone(action).commit(&current_table).await?;
            current_table = Self::apply(
                current_table,
                action_commit,
                &mut existing_updates,
                &mut existing_requirements,
            )?;
        }

        Ok(current_table)
    }
}

/// Parse a `format-version` table-property value into a [`FormatVersion`], matching the domain of
/// Java's `PropertyUtil.propertyAsInt` (`Integer.parseInt`) feeding `Builder.upgradeFormatVersion`
/// (which caps at `SUPPORTED_TABLE_FORMAT_VERSION`): only `1`, `2`, and `3` are legal. Anything
/// else — non-numeric, out of range, negative — is a hard [`ErrorKind::DataInvalid`], never a
/// silent fallback to the existing version.
fn parse_format_version_property(raw: &str) -> Result<FormatVersion> {
    match raw.parse::<u8>() {
        Ok(1) => Ok(FormatVersion::V1),
        Ok(2) => Ok(FormatVersion::V2),
        Ok(3) => Ok(FormatVersion::V3),
        _ => Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Invalid `{}` property value `{raw}` on replace: expected one of 1, 2, 3",
                TableProperties::PROPERTY_FORMAT_VERSION
            ),
        )),
    }
}

#[cfg(test)]
#[path = "staged_table_version_tests.rs"]
mod version_tests;

#[cfg(test)]
#[path = "staged_table_rtas_ops_tests.rs"]
mod rtas_ops_tests;

#[cfg(test)]
#[path = "staged_table_tests.rs"]
mod staged_tests;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::io::{FileIOBuilder, LocalFsStorageFactory};
    use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use crate::spec::{NestedField, PrimitiveType, Schema, Type};
    use crate::{Catalog, CatalogBuilder};

    fn schema_id_name() -> Schema {
        Schema::builder()
            .with_fields(vec![
                Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::required(
                    2,
                    "name",
                    Type::Primitive(PrimitiveType::String),
                )),
            ])
            .build()
            .unwrap()
    }

    async fn shared_fs_catalog(warehouse: &str) -> (impl Catalog, FileIO) {
        // LocalFs so catalog FileIO and staged FileIO share the same on-disk store
        // (MemoryStorageFactory builds a fresh HashMap per FileIO).
        let factory = Arc::new(LocalFsStorageFactory);
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(factory.clone())
            .load(
                "mem",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
            )
            .await
            .unwrap();
        let file_io = FileIOBuilder::new(factory).build();
        (catalog, file_io)
    }

    #[tokio::test]
    async fn replace_invalid_format_version_property_errors_and_keeps_original() {
        // F-1: an unparsable / out-of-range `format-version` property on a staged replace is a hard
        // `DataInvalid` in `parse_format_version_property` — NEVER a silent fallback to the existing
        // version. Without this pin, mutating the guard's invalid-value arm to
        // `_ => Ok(FormatVersion::V2)` (silent fallback) survives the whole suite. Each invalid
        // input must (a) fail `begin_replace` with `ErrorKind::DataInvalid` and (b) leave the
        // original V2 table current & unchanged in the catalog (same guarantee the downgrade test
        // pins). `"2 "` is included precisely because a naive `trim().parse()` would accept it as 2.
        let tmp = TempDir::new().unwrap();
        let warehouse = tmp.path().to_string_lossy().to_string();
        let (catalog, _) = shared_fs_catalog(&warehouse).await;
        let ns = NamespaceIdent::new("sales".into());
        catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
        let table_location = format!("{warehouse}/sales/orders");
        let original = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("orders".into())
                    .location(table_location.clone())
                    .schema(schema_id_name())
                    .format_version(FormatVersion::V2)
                    .build(),
            )
            .await
            .unwrap();
        let original_meta = original.metadata_location_result().unwrap().to_string();

        for value in ["abc", "", "  ", "2 ", "0", "-1", "4", "256"] {
            let creation = TableCreation::builder()
                .name("orders".into())
                .schema(schema_id_name())
                .properties(HashMap::from([(
                    TableProperties::PROPERTY_FORMAT_VERSION.to_string(),
                    value.to_string(),
                )]))
                .build();
            let err = match StagedTableTransaction::begin_replace(&original, creation).await {
                Ok(_) => panic!(
                    "invalid format-version property `{value}` must error, not succeed on replace"
                ),
                Err(e) => e,
            };
            assert_eq!(
                err.kind(),
                ErrorKind::DataInvalid,
                "invalid format-version property `{value}` must be DataInvalid, got: {err}"
            );

            // The original table is still current & unchanged (V2, same metadata pointer) after the
            // rejected replace of value `{value}`.
            let still = catalog.load_table(original.identifier()).await.unwrap();
            assert_eq!(
                still.metadata_location_result().unwrap(),
                original_meta.as_str(),
                "rejected replace of `{value}` moved the catalog metadata pointer"
            );
            assert_eq!(
                still.metadata().format_version(),
                FormatVersion::V2,
                "rejected replace of `{value}` changed the original format version"
            );
        }
    }
}
