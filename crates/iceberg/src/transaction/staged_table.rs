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
//! Java interop battery is a disclosed non-goal of the first unit (GAP_MATRIX 🟡).

use std::str::FromStr;
use std::sync::Arc;

use crate::error::{Error, ErrorKind, Result};
use crate::io::FileIO;
use crate::spec::{DataFile, TableMetadataBuilder};
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
        })
    }

    /// Begin a **replace** transaction against an existing catalog table.
    ///
    /// Builds fresh empty table metadata that keeps the table's **existing root location** (or the
    /// caller-provided `creation.location`). A replace never relocates the table, so repeated
    /// CREATE OR REPLACE cycles leave `metadata().location()` identical every time. The original
    /// catalog entry stays current until [`StagedTableTransaction::commit`]; isolation comes from
    /// deferring the catalog pointer swap, not from a separate on-disk directory.
    pub async fn begin_replace(existing: &Table, mut creation: TableCreation) -> Result<Self> {
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
        // A replace keeps the table's root location STABLE: reuse the caller-provided location if
        // any, else the existing table's current location. Do NOT derive a `__staged_replace`
        // suffix — baking a stage suffix into the metadata relocated the table on every replace and
        // compounded it (orders__staged_replace__staged_replace…), sending future writers to a
        // drifted path and orphaning intent (finding N2). Staging isolation comes from NOT moving
        // the catalog pointer until `commit`, not from a separate directory: the new metadata file
        // gets a fresh version+UUID under the stable location's `metadata/` dir and only becomes
        // current at publish. Data already written elsewhere stays readable — manifests are absolute.
        let table_location = creation.location.clone().unwrap_or_else(|| {
            existing
                .metadata()
                .location()
                .trim_end_matches('/')
                .to_string()
        });
        creation.location = Some(table_location.clone());

        let metadata = TableMetadataBuilder::from_table_creation(creation)?
            .build()?
            .metadata;
        let metadata_location =
            MetadataLocation::new_with_table_location(&table_location).to_string();
        metadata
            .write_to(existing.file_io(), &metadata_location)
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

    /// Apply pending appends to local metadata, write the final metadata file, then publish.
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
        if self.pending_data_files.is_empty() {
            return Ok(self.table);
        }
        let tx = Transaction::new(&self.table);
        let tx = tx
            .fast_append()
            .add_data_files(self.pending_data_files)
            .apply(tx)?;
        tx.apply_locally().await
    }
}

impl Transaction {
    /// Apply all registered actions and write a new metadata file **without** catalog publish.
    ///
    /// Used by [`StagedTableTransaction`]: the engine finishes FileIO work first, then publishes
    /// the pointer in one catalog step.
    pub async fn apply_locally(self) -> Result<Table> {
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

        let current_location = current_table.metadata_location_result()?;
        let next_location = MetadataLocation::from_str(current_location)?
            .with_next_version()
            .to_string();
        current_table
            .metadata()
            .write_to(current_table.file_io(), &next_location)
            .await?;
        Ok(current_table.with_metadata_location(next_location))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::io::{FileIOBuilder, LocalFsStorageFactory, MemoryStorageFactory};
    use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema,
        Struct, Type,
    };
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

    fn data_file(path: &str, records: u64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(records)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .expect("build data file")
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
    async fn create_abort_before_commit_leaves_no_table() {
        let tmp = TempDir::new().unwrap();
        let warehouse = tmp.path().to_string_lossy().to_string();
        let (catalog, file_io) = shared_fs_catalog(&warehouse).await;
        let ns = NamespaceIdent::new("sales".into());
        catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
        let ident = TableIdent::new(ns, "orders".into());
        let location = format!("{warehouse}/sales/orders");
        let creation = TableCreation::builder()
            .name("orders".into())
            .location(location)
            .schema(schema_id_name())
            .build();
        let staged = StagedTableTransaction::begin_create(file_io, ident.clone(), creation)
            .await
            .unwrap();
        assert_eq!(staged.mode(), StagedTableMode::Create);
        drop(staged);
        assert!(!catalog.table_exists(&ident).await.unwrap());
    }

    #[tokio::test]
    async fn create_commit_publishes_once() {
        let tmp = TempDir::new().unwrap();
        let warehouse = tmp.path().to_string_lossy().to_string();
        let (catalog, file_io) = shared_fs_catalog(&warehouse).await;
        let ns = NamespaceIdent::new("sales".into());
        catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
        let ident = TableIdent::new(ns, "orders".into());
        let location = format!("{warehouse}/sales/orders");
        let creation = TableCreation::builder()
            .name("orders".into())
            .location(location.clone())
            .schema(schema_id_name())
            .build();
        let staged = StagedTableTransaction::begin_create(file_io, ident.clone(), creation)
            .await
            .unwrap();
        let file_path = format!("{location}/data/f1.parquet");
        let published = staged
            .add_data_files(vec![data_file(&file_path, 2)])
            .commit(&catalog)
            .await
            .unwrap();
        assert_eq!(published.identifier(), &ident);
        assert!(catalog.table_exists(&ident).await.unwrap());
        let loaded = catalog.load_table(&ident).await.unwrap();
        assert!(loaded.metadata().current_snapshot().is_some());
        assert_eq!(
            loaded
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            crate::spec::Operation::Append
        );
    }

    #[tokio::test]
    async fn create_publish_reload_failure_leaves_no_catalog_entry() {
        // Finding N1: the catalog and the staged writer use SEPARATE in-memory stores
        // (`MemoryStorageFactory` builds a fresh HashMap per FileIO), so the staged metadata is
        // UNREADABLE by the catalog's FileIO — the real staged-CTAS reload failure during publish.
        // Risk pinned: an insert-then-read publish leaves a half-created table (`table_exists`
        // true, `load_table` errors) and breaks `IF NOT EXISTS` idempotency on retry.
        let tmp = TempDir::new().unwrap();
        let warehouse = tmp.path().to_string_lossy().to_string();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "mem",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
            )
            .await
            .expect("load catalog");
        let staged_file_io = FileIOBuilder::new(Arc::new(MemoryStorageFactory)).build();

        let ns = NamespaceIdent::new("sales".into());
        catalog
            .create_namespace(&ns, HashMap::new())
            .await
            .expect("create namespace");
        let ident = TableIdent::new(ns.clone(), "orders".into());
        let location = format!("{warehouse}/sales/orders");
        let creation = TableCreation::builder()
            .name("orders".into())
            .location(location.clone())
            .schema(schema_id_name())
            .build();

        let staged = StagedTableTransaction::begin_create(staged_file_io, ident.clone(), creation)
            .await
            .expect("begin create");

        // Publish MUST error: the catalog's FileIO cannot read the staged metadata.
        staged
            .commit(&catalog)
            .await
            .expect_err("publish must fail when the catalog cannot read the staged metadata");

        // ... and MUST leave no catalog pointer behind (the atomicity guarantee).
        assert!(
            !catalog.table_exists(&ident).await.expect("table_exists"),
            "a failed create-publish must leave no catalog entry (half-created table)"
        );

        // Idempotency intact: the identifier is free, so re-creating the SAME table succeeds
        // (the pre-fix insert-then-read path fails here with TableAlreadyExists).
        let recreated = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("orders".into())
                    .location(location)
                    .schema(schema_id_name())
                    .build(),
            )
            .await
            .expect("re-create of the same identifier after a failed publish must succeed");
        assert_eq!(recreated.identifier(), &ident);
        assert!(
            catalog
                .table_exists(&ident)
                .await
                .expect("table_exists after recreate"),
            "the re-created table must be present"
        );
    }

    #[tokio::test]
    async fn replace_abort_before_commit_keeps_original() {
        let tmp = TempDir::new().unwrap();
        let warehouse = tmp.path().to_string_lossy().to_string();
        let (catalog, _) = shared_fs_catalog(&warehouse).await;
        let ns = NamespaceIdent::new("sales".into());
        catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
        let original = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("orders".into())
                    .schema(schema_id_name())
                    .build(),
            )
            .await
            .unwrap();
        let original_meta = original.metadata_location_result().unwrap().to_string();

        let creation = TableCreation::builder()
            .name("orders".into())
            .schema(schema_id_name())
            .build();
        let staged = StagedTableTransaction::begin_replace(&original, creation)
            .await
            .unwrap();
        let staged =
            staged.add_data_files(vec![data_file(&format!("{warehouse}/stage/f.parquet"), 9)]);
        drop(staged);

        let still = catalog.load_table(original.identifier()).await.unwrap();
        assert_eq!(
            still.metadata_location_result().unwrap(),
            original_meta.as_str()
        );
        assert!(still.metadata().current_snapshot().is_none());
    }

    #[tokio::test]
    async fn replace_commit_swaps_pointer() {
        let tmp = TempDir::new().unwrap();
        let warehouse = tmp.path().to_string_lossy().to_string();
        let (catalog, _) = shared_fs_catalog(&warehouse).await;
        let ns = NamespaceIdent::new("sales".into());
        catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
        let original = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("orders".into())
                    .schema(schema_id_name())
                    .build(),
            )
            .await
            .unwrap();
        let original_meta = original.metadata_location_result().unwrap().to_string();

        let creation = TableCreation::builder()
            .name("orders".into())
            .schema(schema_id_name())
            .build();
        let staged = StagedTableTransaction::begin_replace(&original, creation)
            .await
            .unwrap();
        let published = staged
            .add_data_files(vec![data_file(&format!("{warehouse}/stage/f.parquet"), 2)])
            .commit(&catalog)
            .await
            .unwrap();
        assert_ne!(
            published.metadata_location_result().unwrap(),
            original_meta.as_str()
        );
        let loaded = catalog.load_table(original.identifier()).await.unwrap();
        assert!(loaded.metadata().current_snapshot().is_some());
    }

    #[tokio::test]
    async fn replace_stale_base_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let warehouse = tmp.path().to_string_lossy().to_string();
        let (catalog, _) = shared_fs_catalog(&warehouse).await;
        let ns = NamespaceIdent::new("sales".into());
        catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
        let original = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("orders".into())
                    .schema(schema_id_name())
                    .build(),
            )
            .await
            .unwrap();

        let creation_a = TableCreation::builder()
            .name("orders".into())
            .schema(schema_id_name())
            .build();
        let staged_a = StagedTableTransaction::begin_replace(&original, creation_a)
            .await
            .unwrap();

        let creation_b = TableCreation::builder()
            .name("orders".into())
            .schema(schema_id_name())
            .build();
        let staged_b = StagedTableTransaction::begin_replace(&original, creation_b)
            .await
            .unwrap();
        let _winner = staged_b.commit(&catalog).await.unwrap();

        let err = staged_a.commit(&catalog).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
        assert!(err.retryable(), "stale replace must be retryable: {err}");
    }

    #[tokio::test]
    async fn replace_cycle_keeps_location_stable_and_reads_latest() {
        // Finding N2: a published replace must NOT relocate the table. Across repeated CREATE OR
        // REPLACE cycles `metadata().location()` stays equal to the ORIGINAL location (no
        // `__staged_replace` suffix, no compounding), and each replace's read surface reflects
        // ONLY that replace's data (a replace builds fresh metadata, so `total-records` is this
        // cycle's count, never an accumulation).
        let tmp = TempDir::new().unwrap();
        let warehouse = tmp.path().to_string_lossy().to_string();
        let (catalog, _) = shared_fs_catalog(&warehouse).await;
        let ns = NamespaceIdent::new("sales".into());
        catalog
            .create_namespace(&ns, HashMap::new())
            .await
            .expect("create namespace");

        let table_location = format!("{warehouse}/sales/orders");
        let original = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("orders".into())
                    .location(table_location.clone())
                    .schema(schema_id_name())
                    .build(),
            )
            .await
            .expect("create original");
        assert_eq!(original.metadata().location(), table_location.as_str());

        let mut current = original;
        for (cycle, records) in [11_u64, 22, 33].into_iter().enumerate() {
            let creation = TableCreation::builder()
                .name("orders".into())
                .schema(schema_id_name())
                .build();
            let staged = StagedTableTransaction::begin_replace(&current, creation)
                .await
                .expect("begin replace");
            let data_path = format!("{table_location}/data/cycle-{cycle}.parquet");
            let published = staged
                .add_data_files(vec![data_file(&data_path, records)])
                .commit(&catalog)
                .await
                .expect("publish replace");

            // The published table's root location is the ORIGINAL — every cycle, no drift.
            assert_eq!(
                published.metadata().location(),
                table_location.as_str(),
                "replace cycle {cycle} drifted the table location"
            );

            // The catalog read surface agrees, and exposes exactly THIS replace's data.
            let reloaded = catalog
                .load_table(current.identifier())
                .await
                .expect("reload after replace");
            assert_eq!(
                reloaded.metadata().location(),
                table_location.as_str(),
                "reloaded location drifted at cycle {cycle}"
            );
            let snapshot = reloaded
                .metadata()
                .current_snapshot()
                .expect("current snapshot after replace");
            assert_eq!(
                snapshot
                    .summary()
                    .additional_properties
                    .get("total-records"),
                Some(&records.to_string()),
                "replace cycle {cycle} did not expose the latest data"
            );
            current = reloaded;
        }
    }
}
