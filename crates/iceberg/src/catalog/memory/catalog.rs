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

//! This module contains memory catalog implementation.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::lock::Mutex;
use itertools::Itertools;

use super::namespace_state::NamespaceState;
use crate::catalog::table_metadata_cache::{TableMetadataCache, load_or_fetch_table_metadata};
use crate::io::{FileIO, FileIOBuilder, MemoryStorageFactory, StorageFactory};
use crate::spec::{TableMetadata, TableMetadataBuilder, ViewMetadata, ViewMetadataBuilder};
use crate::table::Table;
use crate::view::{View, ViewCommit};
use crate::{
    Catalog, CatalogBuilder, Error, ErrorKind, MetadataLocation, Namespace, NamespaceIdent, Result,
    TableCommit, TableCreation, TableIdent, ViewCreation,
};

/// Memory catalog warehouse location
pub const MEMORY_CATALOG_WAREHOUSE: &str = "warehouse";

/// namespace `location` property
const LOCATION: &str = "location";

/// Builder for [`MemoryCatalog`].
#[derive(Debug)]
pub struct MemoryCatalogBuilder {
    config: MemoryCatalogConfig,
    storage_factory: Option<Arc<dyn StorageFactory>>,
    /// Opt-in session metadata-pointer cache (FK4.1). Default `None` = OFF.
    table_metadata_cache: Option<Arc<TableMetadataCache>>,
}

impl Default for MemoryCatalogBuilder {
    fn default() -> Self {
        Self {
            config: MemoryCatalogConfig {
                name: None,
                warehouse: "".to_string(),
                props: HashMap::new(),
            },
            storage_factory: None,
            table_metadata_cache: None,
        }
    }
}

impl MemoryCatalogBuilder {
    /// Inject a session-scoped [`TableMetadataCache`] consulted on `load_table`.
    ///
    /// Opt-in, and off by default. The caller owns the `Arc` and may share it across catalogs in
    /// one session. The cache keeps no global state.
    pub fn with_table_metadata_cache(mut self, cache: Arc<TableMetadataCache>) -> Self {
        self.table_metadata_cache = Some(cache);
        self
    }
}

impl CatalogBuilder for MemoryCatalogBuilder {
    type C = MemoryCatalog;

    fn with_storage_factory(mut self, storage_factory: Arc<dyn StorageFactory>) -> Self {
        self.storage_factory = Some(storage_factory);
        self
    }

    fn load(
        mut self,
        name: impl Into<String>,
        props: HashMap<String, String>,
    ) -> impl Future<Output = Result<Self::C>> + Send {
        self.config.name = Some(name.into());

        if props.contains_key(MEMORY_CATALOG_WAREHOUSE) {
            self.config.warehouse = props
                .get(MEMORY_CATALOG_WAREHOUSE)
                .cloned()
                .unwrap_or_default()
        }

        // Collect other remaining properties
        self.config.props = props
            .into_iter()
            .filter(|(k, _)| k != MEMORY_CATALOG_WAREHOUSE)
            .collect();

        let result = {
            if self.config.name.is_none() {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Catalog name is required",
                ))
            } else if self.config.warehouse.is_empty() {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Catalog warehouse is required",
                ))
            } else {
                MemoryCatalog::new(self.config, self.storage_factory, self.table_metadata_cache)
            }
        };

        std::future::ready(result)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MemoryCatalogConfig {
    name: Option<String>,
    warehouse: String,
    props: HashMap<String, String>,
}

/// Memory catalog implementation.
#[derive(Debug)]
pub struct MemoryCatalog {
    name: String,
    root_namespace_state: Mutex<NamespaceState>,
    file_io: FileIO,
    warehouse_location: String,
    properties: HashMap<String, String>,
    /// Opt-in metadata-pointer cache (FK4.1). `None` = default OFF.
    table_metadata_cache: Option<Arc<TableMetadataCache>>,
}

impl MemoryCatalog {
    /// Creates a memory catalog.
    fn new(
        config: MemoryCatalogConfig,
        storage_factory: Option<Arc<dyn StorageFactory>>,
        table_metadata_cache: Option<Arc<TableMetadataCache>>,
    ) -> Result<Self> {
        // Use provided factory or default to MemoryStorageFactory
        let factory = storage_factory.unwrap_or_else(|| Arc::new(MemoryStorageFactory));

        // The builder validates `config.name` is `Some` before constructing the catalog;
        // fall back to the empty name defensively rather than panicking.
        let name = config.name.unwrap_or_default();
        let properties = config.props.clone();

        Ok(Self {
            name,
            root_namespace_state: Mutex::new(NamespaceState::default()),
            file_io: FileIOBuilder::new(factory).with_props(config.props).build(),
            warehouse_location: config.warehouse,
            properties,
            table_metadata_cache,
        })
    }

    /// Snapshot a table's stored metadata location under a short lock (no FileIO).
    async fn table_metadata_location(&self, table_ident: &TableIdent) -> Result<String> {
        let root_namespace_state = self.root_namespace_state.lock().await;
        root_namespace_state
            .get_existing_table_location(table_ident)
            .cloned()
    }

    /// Snapshot a view's stored metadata location under a short lock (no FileIO).
    async fn view_metadata_location(&self, view_ident: &TableIdent) -> Result<String> {
        let root_namespace_state = self.root_namespace_state.lock().await;
        root_namespace_state
            .get_existing_view_location(view_ident)
            .cloned()
    }

    /// Publish parsed metadata into the optional session cache (no-op when cache is OFF).
    fn cache_put(&self, metadata_location: &str, metadata: &TableMetadata) {
        if let Some(cache) = self.table_metadata_cache.as_ref() {
            cache.put(
                metadata_location.to_string(),
                Arc::new(metadata.clone()),
                None,
            );
        }
    }

    /// Load table metadata from FileIO (or the opt-in pointer cache) and assemble a [`Table`].
    /// No catalog lock held.
    async fn load_table_from_location(
        &self,
        table_ident: &TableIdent,
        metadata_location: &str,
    ) -> Result<Table> {
        // v1: location string equality only; MemoryCatalog has no service version token.
        let metadata = load_or_fetch_table_metadata(
            &self.file_io,
            metadata_location,
            self.table_metadata_cache.as_deref(),
            None,
        )
        .await?;

        Table::builder()
            .identifier(table_ident.clone())
            .metadata(metadata)
            .metadata_location(metadata_location.to_string())
            .file_io(self.file_io.clone())
            .build()
    }

    /// Load view metadata from FileIO and assemble a [`View`] — no catalog lock held.
    async fn load_view_from_location(
        &self,
        view_ident: &TableIdent,
        metadata_location: &str,
    ) -> Result<View> {
        let metadata = ViewMetadata::read_from(&self.file_io, metadata_location).await?;

        View::builder()
            .identifier(view_ident.clone())
            .metadata(metadata)
            .metadata_location(metadata_location.to_string())
            .file_io(self.file_io.clone())
            .build()
    }
}

/// Optimistic-concurrency CAS for the in-process catalog. Ports the location-equality check in
/// Java `InMemoryTableOperations.doCommit`, which throws a retryable `CommitFailedException` on a
/// mismatch.
///
/// # Notes
///
/// A stored location equal to the commit base means the commit is current. Anything else is a
/// retryable [`ErrorKind::CatalogCommitConflicts`]. `base_metadata_location == None` models
/// Java's `base == null` create edge. A `None` base never equals a stored location, so it
/// conflicts instead of passing silently. `object_kind` is `"table"` or `"view"`, the only
/// difference between the two Java messages.
fn check_no_concurrent_modification(
    object_kind: &str,
    identifier: &TableIdent,
    stored_metadata_location: &str,
    base_metadata_location: Option<&str>,
    new_metadata_location: &str,
) -> Result<()> {
    if base_metadata_location == Some(stored_metadata_location) {
        return Ok(());
    }

    Err(Error::new(
        ErrorKind::CatalogCommitConflicts,
        format!(
            "Cannot commit to {object_kind} {identifier:?} metadata location from {} to {new_metadata_location} because it has been concurrently modified to {stored_metadata_location}",
            base_metadata_location.unwrap_or("<none>"),
        ),
    )
    .with_retryable(true))
}

#[async_trait]
impl Catalog for MemoryCatalog {
    /// Returns the catalog name supplied at construction (the `name` argument of
    /// [`crate::CatalogBuilder::load`]).
    fn name(&self) -> &str {
        &self.name
    }

    /// Returns the configuration properties supplied at construction.
    fn properties(&self) -> &HashMap<String, String> {
        &self.properties
    }

    /// List namespaces inside the catalog.
    async fn list_namespaces(
        &self,
        maybe_parent: Option<&NamespaceIdent>,
    ) -> Result<Vec<NamespaceIdent>> {
        let root_namespace_state = self.root_namespace_state.lock().await;

        match maybe_parent {
            None => {
                let namespaces = root_namespace_state
                    .list_top_level_namespaces()
                    .into_iter()
                    .map(|str| NamespaceIdent::new(str.to_string()))
                    .collect_vec();

                Ok(namespaces)
            }
            Some(parent_namespace_ident) => {
                let namespaces = root_namespace_state
                    .list_namespaces_under(parent_namespace_ident)?
                    .into_iter()
                    .map(|name| {
                        let mut names = parent_namespace_ident.iter().cloned().collect::<Vec<_>>();
                        names.push(name.to_string());
                        NamespaceIdent::from_vec(names)
                    })
                    .collect::<Result<Vec<_>>>()?;

                Ok(namespaces)
            }
        }
    }

    /// Create a new namespace inside the catalog.
    async fn create_namespace(
        &self,
        namespace_ident: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<Namespace> {
        let mut root_namespace_state = self.root_namespace_state.lock().await;

        root_namespace_state.insert_new_namespace(namespace_ident, properties.clone())?;
        let namespace = Namespace::with_properties(namespace_ident.clone(), properties);

        Ok(namespace)
    }

    /// Get a namespace information from the catalog.
    async fn get_namespace(&self, namespace_ident: &NamespaceIdent) -> Result<Namespace> {
        let root_namespace_state = self.root_namespace_state.lock().await;

        let namespace = Namespace::with_properties(
            namespace_ident.clone(),
            root_namespace_state
                .get_properties(namespace_ident)?
                .clone(),
        );

        Ok(namespace)
    }

    /// Check if namespace exists in catalog.
    async fn namespace_exists(&self, namespace_ident: &NamespaceIdent) -> Result<bool> {
        let guarded_namespaces = self.root_namespace_state.lock().await;

        Ok(guarded_namespaces.namespace_exists(namespace_ident))
    }

    /// Update a namespace inside the catalog.
    ///
    /// # Behavior
    ///
    /// The properties must be the full set of namespace.
    async fn update_namespace(
        &self,
        namespace_ident: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<()> {
        let mut root_namespace_state = self.root_namespace_state.lock().await;

        root_namespace_state.replace_properties(namespace_ident, properties)
    }

    /// Drop a namespace from the catalog.
    async fn drop_namespace(&self, namespace_ident: &NamespaceIdent) -> Result<()> {
        let mut root_namespace_state = self.root_namespace_state.lock().await;

        root_namespace_state.remove_existing_namespace(namespace_ident)
    }

    /// List tables from namespace.
    async fn list_tables(&self, namespace_ident: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        let root_namespace_state = self.root_namespace_state.lock().await;

        let table_names = root_namespace_state.list_tables(namespace_ident)?;
        let table_idents = table_names
            .into_iter()
            .map(|table_name| TableIdent::new(namespace_ident.clone(), table_name.clone()))
            .collect_vec();

        Ok(table_idents)
    }

    /// Create a new table inside the namespace.
    ///
    /// The metadata write runs outside the catalog lock. The pointer insert is a short critical
    /// section after a successful write, so a failed write leaves no catalog entry. Two concurrent
    /// creates of one name race at `insert_new_table`.
    async fn create_table(
        &self,
        namespace_ident: &NamespaceIdent,
        table_creation: TableCreation,
    ) -> Result<Table> {
        let table_name = table_creation.name.clone();
        let table_ident = TableIdent::new(namespace_ident.clone(), table_name);

        // Resolve the table location under a short lock (may need namespace properties).
        let (table_creation, location) = match table_creation.location.clone() {
            Some(location) => (table_creation, location),
            None => {
                let root_namespace_state = self.root_namespace_state.lock().await;
                let namespace_properties = root_namespace_state.get_properties(namespace_ident)?;
                let location_prefix = match namespace_properties.get(LOCATION) {
                    Some(namespace_location) => namespace_location.clone(),
                    None => format!("{}/{}", self.warehouse_location, namespace_ident.join("/")),
                };
                let location = format!("{}/{}", location_prefix, table_ident.name());
                let new_table_creation = TableCreation {
                    location: Some(location.clone()),
                    ..table_creation
                };
                (new_table_creation, location)
            }
        };

        let metadata = TableMetadataBuilder::from_table_creation(table_creation)?
            .build()?
            .metadata;
        let metadata_location = MetadataLocation::new_with_table_location(location).to_string();

        // Write outside the lock so concurrent catalog ops are not serialized on FileIO.
        metadata.write_to(&self.file_io, &metadata_location).await?;

        {
            let mut root_namespace_state = self.root_namespace_state.lock().await;
            root_namespace_state.insert_new_table(&table_ident, metadata_location.clone())?;
        }
        // Seed only after the pointer is claimed — a failed insert must not leave a cache entry
        // for a location the catalog does not own.
        self.cache_put(&metadata_location, &metadata);

        Table::builder()
            .file_io(self.file_io.clone())
            .metadata_location(metadata_location)
            .metadata(metadata)
            .identifier(table_ident)
            .build()
    }

    /// Load table from the catalog.
    ///
    /// Snapshot the metadata pointer under a short lock, then read FileIO outside it, so
    /// concurrent loads and commits do not serialize on metadata I/O.
    ///
    /// With a [`TableMetadataCache`] injected, an unchanged pointer reuses the cached `Arc` and
    /// skips the GET and the re-parse.
    async fn load_table(&self, table_ident: &TableIdent) -> Result<Table> {
        let metadata_location = self.table_metadata_location(table_ident).await?;
        self.load_table_from_location(table_ident, &metadata_location)
            .await
    }

    /// Drop a table from the catalog.
    ///
    /// Removes the pointer under a short lock, then deletes the metadata file outside it.
    /// Evicts the opt-in pointer-cache entry for the dropped location (if a cache is injected).
    async fn drop_table(&self, table_ident: &TableIdent) -> Result<()> {
        let metadata_location = {
            let mut root_namespace_state = self.root_namespace_state.lock().await;
            root_namespace_state.remove_existing_table(table_ident)?
        };
        if let Some(cache) = self.table_metadata_cache.as_ref() {
            cache.invalidate(&metadata_location);
        }
        self.file_io.delete(&metadata_location).await
    }

    /// Check if a table exists in the catalog.
    async fn table_exists(&self, table_ident: &TableIdent) -> Result<bool> {
        let root_namespace_state = self.root_namespace_state.lock().await;

        root_namespace_state.table_exists(table_ident)
    }

    /// Rename a table in the catalog.
    async fn rename_table(
        &self,
        src_table_ident: &TableIdent,
        dst_table_ident: &TableIdent,
    ) -> Result<()> {
        let mut root_namespace_state = self.root_namespace_state.lock().await;

        let mut new_root_namespace_state = root_namespace_state.clone();
        let metadata_location = new_root_namespace_state
            .get_existing_table_location(src_table_ident)?
            .clone();
        new_root_namespace_state.remove_existing_table(src_table_ident)?;
        new_root_namespace_state.insert_new_table(dst_table_ident, metadata_location)?;
        *root_namespace_state = new_root_namespace_state;

        Ok(())
    }

    /// Register an existing table (also the default publish path for a staged **create** via
    /// [`Catalog::publish_create_table`]).
    ///
    /// # Notes
    ///
    /// Registration is all-or-nothing. The read of `metadata_location` proves this catalog's
    /// [`FileIO`] reaches the metadata, and it happens before the pointer insert. The read runs
    /// outside the catalog lock, and the insert is a short critical section after it. A failed
    /// read leaves the catalog unchanged, so a later create of the same identifier succeeds. An
    /// insert-first order would leave a pointer whose `load_table` fails.
    async fn register_table(
        &self,
        table_ident: &TableIdent,
        metadata_location: String,
    ) -> Result<Table> {
        // Read (and validate reachability of) the metadata BEFORE claiming the pointer, so a reload
        // failure cannot leave a half-created table. See the atomicity guarantee above.
        // Goes through the opt-in pointer cache (miss path still fail-closed on unreadable files).
        let metadata = load_or_fetch_table_metadata(
            &self.file_io,
            &metadata_location,
            self.table_metadata_cache.as_deref(),
            None,
        )
        .await?;

        {
            let mut root_namespace_state = self.root_namespace_state.lock().await;
            if let Err(e) =
                root_namespace_state.insert_new_table(table_ident, metadata_location.clone())
            {
                // load_or_fetch may have installed a cache entry; do not keep it if the pointer
                // was never claimed (e.g. table already exists).
                if let Some(cache) = self.table_metadata_cache.as_ref() {
                    cache.invalidate(&metadata_location);
                }
                return Err(e);
            }
        }

        Table::builder()
            .file_io(self.file_io.clone())
            .metadata_location(metadata_location)
            .metadata(metadata)
            .identifier(table_ident.clone())
            .build()
    }

    /// Atomically swap the table's metadata pointer to a fully staged replace (one lock).
    async fn publish_replace_table(
        &self,
        table: Table,
        expected_base_metadata_location: Option<String>,
    ) -> Result<Table> {
        let mut root_namespace_state = self.root_namespace_state.lock().await;
        let ident = table.identifier().clone();
        let stored = root_namespace_state
            .get_existing_table_location(&ident)?
            .clone();
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
        // `commit_table_update` requires the table already exists and overwrites the pointer.
        let updated = root_namespace_state.commit_table_update(table)?;
        Ok(updated)
    }

    /// Update a table in the catalog.
    ///
    /// Optimistic CAS over short critical sections. Step 1 snapshots the stored pointer under the
    /// lock. Step 2 loads, applies, and writes the metadata outside the lock. Step 3 re-reads the
    /// pointer under the lock, compares it with the commit base, and flips it on a match.
    ///
    /// A concurrent winner advances the stored location, so step 3 returns a retryable
    /// [`ErrorKind::CatalogCommitConflicts`]. FileIO never runs under the lock.
    async fn update_table(&self, commit: TableCommit) -> Result<Table> {
        let table_ident = commit.identifier().clone();
        let base_metadata_location = commit.base_metadata_location().map(str::to_string);

        // 1. Snapshot pointer under a short lock.
        let stored_at_start = self.table_metadata_location(&table_ident).await?;

        // 2. Load + apply + write outside the lock.
        let current_table = self
            .load_table_from_location(&table_ident, &stored_at_start)
            .await?;
        let staged_table = commit.apply(current_table)?;
        let new_metadata_location = staged_table.metadata_location_result()?.to_string();

        // Early CAS against the load snapshot — cheap reject when the commit was already stale
        // at start (no FileIO).
        check_no_concurrent_modification(
            "table",
            staged_table.identifier(),
            &stored_at_start,
            base_metadata_location.as_deref(),
            &new_metadata_location,
        )?;

        // Recheck the pointer under a short lock before the write. A concurrent winner that
        // advanced it during load and apply makes this refuse, and no orphan file is written. A
        // loser that races between this recheck and the step 3 CAS may still leave one orphan
        // file. The pointer never half-flips.
        {
            let root_namespace_state = self.root_namespace_state.lock().await;
            let stored_mid = root_namespace_state
                .get_existing_table_location(&table_ident)?
                .clone();
            check_no_concurrent_modification(
                "table",
                staged_table.identifier(),
                &stored_mid,
                base_metadata_location.as_deref(),
                &new_metadata_location,
            )?;
        }

        staged_table
            .metadata()
            .write_to(staged_table.file_io(), &new_metadata_location)
            .await?;

        // 3. Authoritative pointer CAS under a short lock (I/O already complete).
        let mut root_namespace_state = self.root_namespace_state.lock().await;
        let stored_now = root_namespace_state
            .get_existing_table_location(&table_ident)?
            .clone();
        check_no_concurrent_modification(
            "table",
            staged_table.identifier(),
            &stored_now,
            base_metadata_location.as_deref(),
            &new_metadata_location,
        )?;
        let updated_table = root_namespace_state.commit_table_update(staged_table)?;
        // Evict the prior pointer (limit session retention) and seed the new one so the next
        // load_table is a hit without re-GET.
        if let Some(cache) = self.table_metadata_cache.as_ref()
            && stored_at_start != new_metadata_location
        {
            cache.invalidate(&stored_at_start);
        }
        self.cache_put(
            updated_table
                .metadata_location()
                .unwrap_or(new_metadata_location.as_str()),
            updated_table.metadata(),
        );

        Ok(updated_table)
    }

    /// Evict the cached entry for this table's metadata location. A no-op with no cache, or with
    /// no such table.
    ///
    /// Mirrors Java `Catalog.invalidateTable`. An unknown table leaves other keys alone. A
    /// `clear()` of the whole cache would thrash unrelated entries.
    async fn invalidate_table(&self, table: &TableIdent) -> Result<()> {
        let Some(cache) = self.table_metadata_cache.as_ref() else {
            return Ok(());
        };
        if let Ok(location) = self.table_metadata_location(table).await {
            cache.invalidate(&location);
        }
        Ok(())
    }

    async fn list_views(&self, namespace_ident: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        let root_namespace_state = self.root_namespace_state.lock().await;

        let view_names = root_namespace_state.list_views(namespace_ident)?;

        let views = view_names
            .into_iter()
            .map(|view_name| TableIdent::new(namespace_ident.clone(), view_name.clone()))
            .collect_vec();

        Ok(views)
    }

    async fn create_view(
        &self,
        namespace_ident: &NamespaceIdent,
        view_creation: ViewCreation,
    ) -> Result<View> {
        let view_name = view_creation.name.clone();
        let view_ident = TableIdent::new(namespace_ident.clone(), view_name);
        let location = view_creation.location.clone();

        let metadata = ViewMetadataBuilder::from_view_creation(view_creation)?
            .build()?
            .metadata;
        let metadata_location = MetadataLocation::new_with_table_location(location).to_string();

        // Write outside the lock (same half-create refusal as create_table: insert only after write).
        metadata.write_to(&self.file_io, &metadata_location).await?;

        {
            let mut root_namespace_state = self.root_namespace_state.lock().await;
            root_namespace_state.insert_new_view(&view_ident, metadata_location.clone())?;
        }

        View::builder()
            .file_io(self.file_io.clone())
            .metadata_location(metadata_location)
            .metadata(metadata)
            .identifier(view_ident)
            .build()
    }

    async fn load_view(&self, view_ident: &TableIdent) -> Result<View> {
        let metadata_location = self.view_metadata_location(view_ident).await?;
        self.load_view_from_location(view_ident, &metadata_location)
            .await
    }

    async fn drop_view(&self, view_ident: &TableIdent) -> Result<()> {
        let mut root_namespace_state = self.root_namespace_state.lock().await;

        let _ = root_namespace_state.remove_existing_view(view_ident)?;

        Ok(())
    }

    async fn view_exists(&self, view_ident: &TableIdent) -> Result<bool> {
        let root_namespace_state = self.root_namespace_state.lock().await;

        root_namespace_state.view_exists(view_ident)
    }

    async fn rename_view(
        &self,
        src_view_ident: &TableIdent,
        dst_view_ident: &TableIdent,
    ) -> Result<()> {
        let mut root_namespace_state = self.root_namespace_state.lock().await;

        let mut new_root_namespace_state = root_namespace_state.clone();
        let metadata_location = new_root_namespace_state
            .get_existing_view_location(src_view_ident)?
            .clone();
        new_root_namespace_state.remove_existing_view(src_view_ident)?;
        new_root_namespace_state.insert_new_view(dst_view_ident, metadata_location)?;
        *root_namespace_state = new_root_namespace_state;

        Ok(())
    }

    async fn update_view(&self, commit: ViewCommit) -> Result<View> {
        let view_ident = commit.identifier().clone();
        let base_metadata_location = commit.base_metadata_location().map(str::to_string);

        // 1. Snapshot pointer under a short lock.
        let stored_at_start = self.view_metadata_location(&view_ident).await?;

        // 2. Load + apply + write outside the lock.
        let current_view = self
            .load_view_from_location(&view_ident, &stored_at_start)
            .await?;
        let staged_view = commit.apply(current_view)?;
        let new_metadata_location = staged_view.metadata_location_result()?.to_string();

        // Early CAS against the load snapshot — cheap reject when already stale at start.
        check_no_concurrent_modification(
            "view",
            staged_view.identifier(),
            &stored_at_start,
            base_metadata_location.as_deref(),
            &new_metadata_location,
        )?;

        // Mid-point recheck under a short lock BEFORE writing (same orphan-reduction as tables).
        {
            let root_namespace_state = self.root_namespace_state.lock().await;
            let stored_mid = root_namespace_state
                .get_existing_view_location(&view_ident)?
                .clone();
            check_no_concurrent_modification(
                "view",
                staged_view.identifier(),
                &stored_mid,
                base_metadata_location.as_deref(),
                &new_metadata_location,
            )?;
        }

        staged_view
            .metadata()
            .write_to(staged_view.file_io(), &new_metadata_location)
            .await?;

        // 3. Authoritative pointer CAS under a short lock.
        let mut root_namespace_state = self.root_namespace_state.lock().await;
        let stored_now = root_namespace_state
            .get_existing_view_location(&view_ident)?
            .clone();
        check_no_concurrent_modification(
            "view",
            staged_view.identifier(),
            &stored_now,
            base_metadata_location.as_deref(),
            &new_metadata_location,
        )?;
        root_namespace_state.commit_view_update(staged_view.identifier(), new_metadata_location)?;

        Ok(staged_view)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashSet;
    use std::hash::Hash;
    use std::iter::FromIterator;
    use std::vec;

    use regex::Regex;
    use tempfile::TempDir;

    use super::*;
    use crate::io::FileIO;
    use crate::spec::{NestedField, PartitionSpec, PrimitiveType, Schema, SortOrder, Type};
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{TableUpdate, UNNAMED_CATALOG};

    fn temp_path() -> String {
        let temp_dir = TempDir::new().unwrap();
        temp_dir.path().to_str().unwrap().to_string()
    }

    pub(crate) async fn new_memory_catalog() -> impl Catalog {
        let warehouse_location = temp_path();
        MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_location)]),
            )
            .await
            .unwrap()
    }

    async fn create_namespace<C: Catalog>(catalog: &C, namespace_ident: &NamespaceIdent) {
        let _ = catalog
            .create_namespace(namespace_ident, HashMap::new())
            .await
            .unwrap();
    }

    async fn create_namespaces<C: Catalog>(catalog: &C, namespace_idents: &Vec<&NamespaceIdent>) {
        for namespace_ident in namespace_idents {
            let _ = create_namespace(catalog, namespace_ident).await;
        }
    }

    fn to_set<T: Eq + Hash>(vec: Vec<T>) -> HashSet<T> {
        HashSet::from_iter(vec)
    }

    fn simple_table_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap()
    }

    async fn create_table<C: Catalog>(catalog: &C, table_ident: &TableIdent) -> Table {
        catalog
            .create_table(
                &table_ident.namespace,
                TableCreation::builder()
                    .name(table_ident.name().into())
                    .schema(simple_table_schema())
                    .build(),
            )
            .await
            .unwrap()
    }

    async fn create_tables<C: Catalog>(catalog: &C, table_idents: Vec<&TableIdent>) {
        for table_ident in table_idents {
            create_table(catalog, table_ident).await;
        }
    }

    async fn create_table_with_namespace<C: Catalog>(catalog: &C) -> Table {
        let namespace_ident = NamespaceIdent::new("abc".into());
        create_namespace(catalog, &namespace_ident).await;

        let table_ident = TableIdent::new(namespace_ident, "test".to_string());
        create_table(catalog, &table_ident).await
    }

    fn assert_table_eq(table: &Table, expected_table_ident: &TableIdent, expected_schema: &Schema) {
        assert_eq!(table.identifier(), expected_table_ident);

        let metadata = table.metadata();

        assert_eq!(metadata.current_schema().as_ref(), expected_schema);

        let expected_partition_spec = PartitionSpec::builder((*expected_schema).clone())
            .with_spec_id(0)
            .build()
            .unwrap();

        assert_eq!(
            metadata
                .partition_specs_iter()
                .map(|p| p.as_ref())
                .collect_vec(),
            vec![&expected_partition_spec]
        );

        let expected_sorted_order = SortOrder::builder()
            .with_order_id(0)
            .with_fields(vec![])
            .build(expected_schema)
            .unwrap();

        assert_eq!(
            metadata
                .sort_orders_iter()
                .map(|s| s.as_ref())
                .collect_vec(),
            vec![&expected_sorted_order]
        );

        assert_eq!(metadata.properties(), &HashMap::new());

        assert!(!table.readonly());
    }

    const UUID_REGEX_STR: &str = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";

    fn assert_table_metadata_location_matches(table: &Table, regex_str: &str) {
        let actual = table.metadata_location().unwrap().to_string();
        let regex = Regex::new(regex_str).unwrap();
        assert!(
            regex.is_match(&actual),
            "Expected metadata location to match regex, but got location: {actual} and regex: {regex}"
        )
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_empty_vector() {
        let catalog = new_memory_catalog().await;

        assert_eq!(catalog.list_namespaces(None).await.unwrap(), vec![]);
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_single_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("abc".into());
        create_namespace(&catalog, &namespace_ident).await;

        assert_eq!(catalog.list_namespaces(None).await.unwrap(), vec![
            namespace_ident
        ]);
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_multiple_namespaces() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_1 = NamespaceIdent::new("a".into());
        let namespace_ident_2 = NamespaceIdent::new("b".into());
        create_namespaces(&catalog, &vec![&namespace_ident_1, &namespace_ident_2]).await;

        assert_eq!(
            to_set(catalog.list_namespaces(None).await.unwrap()),
            to_set(vec![namespace_ident_1, namespace_ident_2])
        );
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_only_top_level_namespaces() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_1 = NamespaceIdent::new("a".into());
        let namespace_ident_2 = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_3 = NamespaceIdent::new("b".into());
        create_namespaces(&catalog, &vec![
            &namespace_ident_1,
            &namespace_ident_2,
            &namespace_ident_3,
        ])
        .await;

        assert_eq!(
            to_set(catalog.list_namespaces(None).await.unwrap()),
            to_set(vec![namespace_ident_1, namespace_ident_3])
        );
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_no_namespaces_under_parent() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_1 = NamespaceIdent::new("a".into());
        let namespace_ident_2 = NamespaceIdent::new("b".into());
        create_namespaces(&catalog, &vec![&namespace_ident_1, &namespace_ident_2]).await;

        assert_eq!(
            catalog
                .list_namespaces(Some(&namespace_ident_1))
                .await
                .unwrap(),
            vec![]
        );
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_namespace_under_parent() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_1 = NamespaceIdent::new("a".into());
        let namespace_ident_2 = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_3 = NamespaceIdent::new("c".into());
        create_namespaces(&catalog, &vec![
            &namespace_ident_1,
            &namespace_ident_2,
            &namespace_ident_3,
        ])
        .await;

        assert_eq!(
            to_set(catalog.list_namespaces(None).await.unwrap()),
            to_set(vec![namespace_ident_1.clone(), namespace_ident_3])
        );

        assert_eq!(
            catalog
                .list_namespaces(Some(&namespace_ident_1))
                .await
                .unwrap(),
            vec![namespace_ident_2]
        );
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_multiple_namespaces_under_parent() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_1 = NamespaceIdent::new("a".to_string());
        let namespace_ident_2 = NamespaceIdent::from_strs(vec!["a", "a"]).unwrap();
        let namespace_ident_3 = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_4 = NamespaceIdent::from_strs(vec!["a", "c"]).unwrap();
        let namespace_ident_5 = NamespaceIdent::new("b".into());
        create_namespaces(&catalog, &vec![
            &namespace_ident_1,
            &namespace_ident_2,
            &namespace_ident_3,
            &namespace_ident_4,
            &namespace_ident_5,
        ])
        .await;

        assert_eq!(
            to_set(
                catalog
                    .list_namespaces(Some(&namespace_ident_1))
                    .await
                    .unwrap()
            ),
            to_set(vec![
                namespace_ident_2,
                namespace_ident_3,
                namespace_ident_4,
            ])
        );
    }

    #[tokio::test]
    async fn test_namespace_exists_returns_false() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;

        assert!(
            !catalog
                .namespace_exists(&NamespaceIdent::new("b".into()))
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_namespace_exists_returns_true() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;

        assert!(catalog.namespace_exists(&namespace_ident).await.unwrap());
    }

    #[tokio::test]
    async fn test_create_namespace_with_empty_properties() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("a".into());

        assert_eq!(
            catalog
                .create_namespace(&namespace_ident, HashMap::new())
                .await
                .unwrap(),
            Namespace::new(namespace_ident.clone())
        );

        assert_eq!(
            catalog.get_namespace(&namespace_ident).await.unwrap(),
            Namespace::with_properties(namespace_ident, HashMap::new())
        );
    }

    #[tokio::test]
    async fn test_create_namespace_with_properties() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("abc".into());

        let mut properties: HashMap<String, String> = HashMap::new();
        properties.insert("k".into(), "v".into());

        assert_eq!(
            catalog
                .create_namespace(&namespace_ident, properties.clone())
                .await
                .unwrap(),
            Namespace::with_properties(namespace_ident.clone(), properties.clone())
        );

        assert_eq!(
            catalog.get_namespace(&namespace_ident).await.unwrap(),
            Namespace::with_properties(namespace_ident, properties)
        );
    }

    #[tokio::test]
    async fn test_create_namespace_throws_error_if_namespace_already_exists() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;

        assert_eq!(
            catalog
                .create_namespace(&namespace_ident, HashMap::new())
                .await
                .unwrap_err()
                .to_string(),
            format!(
                "NamespaceAlreadyExists => Cannot create namespace {:?}. Namespace already exists.",
                &namespace_ident
            )
        );

        assert_eq!(
            catalog.get_namespace(&namespace_ident).await.unwrap(),
            Namespace::with_properties(namespace_ident, HashMap::new())
        );
    }

    #[tokio::test]
    async fn test_create_nested_namespace() {
        let catalog = new_memory_catalog().await;
        let parent_namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &parent_namespace_ident).await;

        let child_namespace_ident = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();

        assert_eq!(
            catalog
                .create_namespace(&child_namespace_ident, HashMap::new())
                .await
                .unwrap(),
            Namespace::new(child_namespace_ident.clone())
        );

        assert_eq!(
            catalog.get_namespace(&child_namespace_ident).await.unwrap(),
            Namespace::with_properties(child_namespace_ident, HashMap::new())
        );
    }

    #[tokio::test]
    async fn test_create_deeply_nested_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespaces(&catalog, &vec![&namespace_ident_a, &namespace_ident_a_b]).await;

        let namespace_ident_a_b_c = NamespaceIdent::from_strs(vec!["a", "b", "c"]).unwrap();

        assert_eq!(
            catalog
                .create_namespace(&namespace_ident_a_b_c, HashMap::new())
                .await
                .unwrap(),
            Namespace::new(namespace_ident_a_b_c.clone())
        );

        assert_eq!(
            catalog.get_namespace(&namespace_ident_a_b_c).await.unwrap(),
            Namespace::with_properties(namespace_ident_a_b_c, HashMap::new())
        );
    }

    #[tokio::test]
    async fn test_create_nested_namespace_throws_error_if_top_level_namespace_doesnt_exist() {
        let catalog = new_memory_catalog().await;

        let nested_namespace_ident = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();

        assert_eq!(
            catalog
                .create_namespace(&nested_namespace_ident, HashMap::new())
                .await
                .unwrap_err()
                .to_string(),
            format!(
                "NamespaceNotFound => No such namespace: {:?}",
                NamespaceIdent::new("a".into())
            )
        );

        assert_eq!(catalog.list_namespaces(None).await.unwrap(), vec![]);
    }

    #[tokio::test]
    async fn test_create_deeply_nested_namespace_throws_error_if_intermediate_namespace_doesnt_exist()
     {
        let catalog = new_memory_catalog().await;

        let namespace_ident_a = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident_a).await;

        let namespace_ident_a_b_c = NamespaceIdent::from_strs(vec!["a", "b", "c"]).unwrap();

        assert_eq!(
            catalog
                .create_namespace(&namespace_ident_a_b_c, HashMap::new())
                .await
                .unwrap_err()
                .to_string(),
            format!(
                "NamespaceNotFound => No such namespace: {:?}",
                NamespaceIdent::from_strs(vec!["a", "b"]).unwrap()
            )
        );

        assert_eq!(catalog.list_namespaces(None).await.unwrap(), vec![
            namespace_ident_a.clone()
        ]);

        assert_eq!(
            catalog
                .list_namespaces(Some(&namespace_ident_a))
                .await
                .unwrap(),
            vec![]
        );
    }

    #[tokio::test]
    async fn test_get_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("abc".into());

        let mut properties: HashMap<String, String> = HashMap::new();
        properties.insert("k".into(), "v".into());
        let _ = catalog
            .create_namespace(&namespace_ident, properties.clone())
            .await
            .unwrap();

        assert_eq!(
            catalog.get_namespace(&namespace_ident).await.unwrap(),
            Namespace::with_properties(namespace_ident, properties)
        )
    }

    #[tokio::test]
    async fn test_get_nested_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespaces(&catalog, &vec![&namespace_ident_a, &namespace_ident_a_b]).await;

        assert_eq!(
            catalog.get_namespace(&namespace_ident_a_b).await.unwrap(),
            Namespace::with_properties(namespace_ident_a_b, HashMap::new())
        );
    }

    #[tokio::test]
    async fn test_get_deeply_nested_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_a_b_c = NamespaceIdent::from_strs(vec!["a", "b", "c"]).unwrap();
        create_namespaces(&catalog, &vec![
            &namespace_ident_a,
            &namespace_ident_a_b,
            &namespace_ident_a_b_c,
        ])
        .await;

        assert_eq!(
            catalog.get_namespace(&namespace_ident_a_b_c).await.unwrap(),
            Namespace::with_properties(namespace_ident_a_b_c, HashMap::new())
        );
    }

    #[tokio::test]
    async fn test_get_namespace_throws_error_if_namespace_doesnt_exist() {
        let catalog = new_memory_catalog().await;
        create_namespace(&catalog, &NamespaceIdent::new("a".into())).await;

        let non_existent_namespace_ident = NamespaceIdent::new("b".into());
        assert_eq!(
            catalog
                .get_namespace(&non_existent_namespace_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("NamespaceNotFound => No such namespace: {non_existent_namespace_ident:?}")
        )
    }

    #[tokio::test]
    async fn test_update_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("abc".into());
        create_namespace(&catalog, &namespace_ident).await;

        let mut new_properties: HashMap<String, String> = HashMap::new();
        new_properties.insert("k".into(), "v".into());

        catalog
            .update_namespace(&namespace_ident, new_properties.clone())
            .await
            .unwrap();

        assert_eq!(
            catalog.get_namespace(&namespace_ident).await.unwrap(),
            Namespace::with_properties(namespace_ident, new_properties)
        )
    }

    #[tokio::test]
    async fn test_update_nested_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespaces(&catalog, &vec![&namespace_ident_a, &namespace_ident_a_b]).await;

        let mut new_properties = HashMap::new();
        new_properties.insert("k".into(), "v".into());

        catalog
            .update_namespace(&namespace_ident_a_b, new_properties.clone())
            .await
            .unwrap();

        assert_eq!(
            catalog.get_namespace(&namespace_ident_a_b).await.unwrap(),
            Namespace::with_properties(namespace_ident_a_b, new_properties)
        );
    }

    #[tokio::test]
    async fn test_update_deeply_nested_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_a_b_c = NamespaceIdent::from_strs(vec!["a", "b", "c"]).unwrap();
        create_namespaces(&catalog, &vec![
            &namespace_ident_a,
            &namespace_ident_a_b,
            &namespace_ident_a_b_c,
        ])
        .await;

        let mut new_properties = HashMap::new();
        new_properties.insert("k".into(), "v".into());

        catalog
            .update_namespace(&namespace_ident_a_b_c, new_properties.clone())
            .await
            .unwrap();

        assert_eq!(
            catalog.get_namespace(&namespace_ident_a_b_c).await.unwrap(),
            Namespace::with_properties(namespace_ident_a_b_c, new_properties)
        );
    }

    #[tokio::test]
    async fn test_update_namespace_throws_error_if_namespace_doesnt_exist() {
        let catalog = new_memory_catalog().await;
        create_namespace(&catalog, &NamespaceIdent::new("abc".into())).await;

        let non_existent_namespace_ident = NamespaceIdent::new("def".into());
        assert_eq!(
            catalog
                .update_namespace(&non_existent_namespace_ident, HashMap::new())
                .await
                .unwrap_err()
                .to_string(),
            format!("NamespaceNotFound => No such namespace: {non_existent_namespace_ident:?}")
        )
    }

    #[tokio::test]
    async fn test_update_namespace_properties_set_only() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("abc".into());
        catalog
            .create_namespace(
                &namespace_ident,
                HashMap::from([("keep".into(), "0".into())]),
            )
            .await
            .unwrap();

        catalog
            .set_namespace_properties(
                &namespace_ident,
                HashMap::from([("k1".into(), "v1".into()), ("k2".into(), "v2".into())]),
            )
            .await
            .unwrap();

        let props = catalog
            .get_namespace(&namespace_ident)
            .await
            .unwrap()
            .properties()
            .clone();
        assert_eq!(props.get("keep"), Some(&"0".to_string()));
        assert_eq!(props.get("k1"), Some(&"v1".to_string()));
        assert_eq!(props.get("k2"), Some(&"v2".to_string()));
    }

    #[tokio::test]
    async fn test_update_namespace_properties_remove_only() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("abc".into());
        catalog
            .create_namespace(
                &namespace_ident,
                HashMap::from([("drop_me".into(), "x".into()), ("keep".into(), "y".into())]),
            )
            .await
            .unwrap();

        catalog
            .remove_namespace_properties(&namespace_ident, HashSet::from(["drop_me".to_string()]))
            .await
            .unwrap();

        let props = catalog
            .get_namespace(&namespace_ident)
            .await
            .unwrap()
            .properties()
            .clone();
        // Mutation guard: dropping the `properties.remove(key)` step in
        // `update_namespace_properties` leaves "drop_me" present and fails this assertion.
        assert_eq!(props.get("drop_me"), None);
        assert_eq!(props.get("keep"), Some(&"y".to_string()));
    }

    #[tokio::test]
    async fn test_update_namespace_properties_set_and_remove_combined() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("abc".into());
        catalog
            .create_namespace(
                &namespace_ident,
                HashMap::from([("drop_me".into(), "x".into()), ("keep".into(), "y".into())]),
            )
            .await
            .unwrap();

        catalog
            .update_namespace_properties(
                &namespace_ident,
                HashSet::from(["drop_me".to_string()]),
                HashMap::from([("new".into(), "z".into())]),
            )
            .await
            .unwrap();

        let props = catalog
            .get_namespace(&namespace_ident)
            .await
            .unwrap()
            .properties()
            .clone();
        assert_eq!(props.get("drop_me"), None);
        assert_eq!(props.get("keep"), Some(&"y".to_string()));
        assert_eq!(props.get("new"), Some(&"z".to_string()));
    }

    #[tokio::test]
    async fn test_update_namespace_properties_overlap_rejected() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("abc".into());
        create_namespace(&catalog, &namespace_ident).await;

        let err = catalog
            .update_namespace_properties(
                &namespace_ident,
                HashSet::from(["dup".to_string()]),
                HashMap::from([("dup".to_string(), "v".to_string())]),
            )
            .await
            .unwrap_err();

        // Mutation guard: dropping the overlap check lets this succeed and fails the assertion.
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    #[tokio::test]
    async fn test_update_namespace_properties_no_such_namespace() {
        let catalog = new_memory_catalog().await;
        let non_existent_namespace_ident = NamespaceIdent::new("def".into());

        let err = catalog
            .update_namespace_properties(
                &non_existent_namespace_ident,
                HashSet::new(),
                HashMap::from([("k".to_string(), "v".to_string())]),
            )
            .await
            .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::NamespaceNotFound);
    }

    #[tokio::test]
    async fn test_update_namespace_properties_remove_missing_key_is_noop() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("abc".into());
        catalog
            .create_namespace(
                &namespace_ident,
                HashMap::from([("keep".into(), "y".into())]),
            )
            .await
            .unwrap();

        // Removing an absent key must not error (Java `removeProperties` tolerance).
        catalog
            .remove_namespace_properties(
                &namespace_ident,
                HashSet::from(["never_existed".to_string()]),
            )
            .await
            .unwrap();

        let props = catalog
            .get_namespace(&namespace_ident)
            .await
            .unwrap()
            .properties()
            .clone();
        assert_eq!(props.get("keep"), Some(&"y".to_string()));
    }

    #[tokio::test]
    async fn test_drop_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("abc".into());
        create_namespace(&catalog, &namespace_ident).await;

        catalog.drop_namespace(&namespace_ident).await.unwrap();

        assert!(!catalog.namespace_exists(&namespace_ident).await.unwrap())
    }

    #[tokio::test]
    async fn test_drop_nested_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespaces(&catalog, &vec![&namespace_ident_a, &namespace_ident_a_b]).await;

        catalog.drop_namespace(&namespace_ident_a_b).await.unwrap();

        assert!(
            !catalog
                .namespace_exists(&namespace_ident_a_b)
                .await
                .unwrap()
        );

        assert!(catalog.namespace_exists(&namespace_ident_a).await.unwrap());
    }

    #[tokio::test]
    async fn test_drop_deeply_nested_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_a_b_c = NamespaceIdent::from_strs(vec!["a", "b", "c"]).unwrap();
        create_namespaces(&catalog, &vec![
            &namespace_ident_a,
            &namespace_ident_a_b,
            &namespace_ident_a_b_c,
        ])
        .await;

        catalog
            .drop_namespace(&namespace_ident_a_b_c)
            .await
            .unwrap();

        assert!(
            !catalog
                .namespace_exists(&namespace_ident_a_b_c)
                .await
                .unwrap()
        );

        assert!(
            catalog
                .namespace_exists(&namespace_ident_a_b)
                .await
                .unwrap()
        );

        assert!(catalog.namespace_exists(&namespace_ident_a).await.unwrap());
    }

    #[tokio::test]
    async fn test_drop_namespace_throws_error_if_namespace_doesnt_exist() {
        let catalog = new_memory_catalog().await;

        let non_existent_namespace_ident = NamespaceIdent::new("abc".into());
        assert_eq!(
            catalog
                .drop_namespace(&non_existent_namespace_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("NamespaceNotFound => No such namespace: {non_existent_namespace_ident:?}")
        )
    }

    #[tokio::test]
    async fn test_drop_namespace_throws_error_if_nested_namespace_doesnt_exist() {
        let catalog = new_memory_catalog().await;
        create_namespace(&catalog, &NamespaceIdent::new("a".into())).await;

        let non_existent_namespace_ident =
            NamespaceIdent::from_vec(vec!["a".into(), "b".into()]).unwrap();
        assert_eq!(
            catalog
                .drop_namespace(&non_existent_namespace_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("NamespaceNotFound => No such namespace: {non_existent_namespace_ident:?}")
        )
    }

    #[tokio::test]
    async fn test_dropping_a_namespace_also_drops_namespaces_nested_under_that_one() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespaces(&catalog, &vec![&namespace_ident_a, &namespace_ident_a_b]).await;

        catalog.drop_namespace(&namespace_ident_a).await.unwrap();

        assert!(!catalog.namespace_exists(&namespace_ident_a).await.unwrap());

        assert!(
            !catalog
                .namespace_exists(&namespace_ident_a_b)
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_create_table_with_location() {
        let tmp_dir = TempDir::new().unwrap();
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;

        let table_name = "abc";
        let location = tmp_dir.path().to_str().unwrap().to_string();
        let table_creation = TableCreation::builder()
            .name(table_name.into())
            .location(location.clone())
            .schema(simple_table_schema())
            .build();

        let expected_table_ident = TableIdent::new(namespace_ident.clone(), table_name.into());

        assert_table_eq(
            &catalog
                .create_table(&namespace_ident, table_creation)
                .await
                .unwrap(),
            &expected_table_ident,
            &simple_table_schema(),
        );

        let table = catalog.load_table(&expected_table_ident).await.unwrap();

        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());

        assert!(
            table
                .metadata_location()
                .unwrap()
                .to_string()
                .starts_with(&location)
        )
    }

    #[tokio::test]
    async fn test_create_table_falls_back_to_namespace_location_if_table_location_is_missing() {
        let warehouse_location = temp_path();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    warehouse_location.clone(),
                )]),
            )
            .await
            .unwrap();

        let namespace_ident = NamespaceIdent::new("a".into());
        let mut namespace_properties = HashMap::new();
        let namespace_location = temp_path();
        namespace_properties.insert(LOCATION.to_string(), namespace_location.to_string());
        catalog
            .create_namespace(&namespace_ident, namespace_properties)
            .await
            .unwrap();

        let table_name = "tbl1";
        let expected_table_ident = TableIdent::new(namespace_ident.clone(), table_name.into());
        let expected_table_metadata_location_regex =
            format!("^{namespace_location}/tbl1/metadata/00000-{UUID_REGEX_STR}.metadata.json$",);

        let table = catalog
            .create_table(
                &namespace_ident,
                TableCreation::builder()
                    .name(table_name.into())
                    .schema(simple_table_schema())
                    // no location specified for table
                    .build(),
            )
            .await
            .unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);

        let table = catalog.load_table(&expected_table_ident).await.unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);
    }

    #[tokio::test]
    async fn test_create_table_in_nested_namespace_falls_back_to_nested_namespace_location_if_table_location_is_missing()
     {
        let warehouse_location = temp_path();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    warehouse_location.clone(),
                )]),
            )
            .await
            .unwrap();

        let namespace_ident = NamespaceIdent::new("a".into());
        let mut namespace_properties = HashMap::new();
        let namespace_location = temp_path();
        namespace_properties.insert(LOCATION.to_string(), namespace_location.to_string());
        catalog
            .create_namespace(&namespace_ident, namespace_properties)
            .await
            .unwrap();

        let nested_namespace_ident = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let mut nested_namespace_properties = HashMap::new();
        let nested_namespace_location = temp_path();
        nested_namespace_properties
            .insert(LOCATION.to_string(), nested_namespace_location.to_string());
        catalog
            .create_namespace(&nested_namespace_ident, nested_namespace_properties)
            .await
            .unwrap();

        let table_name = "tbl1";
        let expected_table_ident =
            TableIdent::new(nested_namespace_ident.clone(), table_name.into());
        let expected_table_metadata_location_regex = format!(
            "^{nested_namespace_location}/tbl1/metadata/00000-{UUID_REGEX_STR}.metadata.json$",
        );

        let table = catalog
            .create_table(
                &nested_namespace_ident,
                TableCreation::builder()
                    .name(table_name.into())
                    .schema(simple_table_schema())
                    // no location specified for table
                    .build(),
            )
            .await
            .unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);

        let table = catalog.load_table(&expected_table_ident).await.unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);
    }

    #[tokio::test]
    async fn test_create_table_falls_back_to_warehouse_location_if_both_table_location_and_namespace_location_are_missing()
     {
        let warehouse_location = temp_path();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    warehouse_location.clone(),
                )]),
            )
            .await
            .unwrap();

        let namespace_ident = NamespaceIdent::new("a".into());
        // note: no location specified in namespace_properties
        let namespace_properties = HashMap::new();
        catalog
            .create_namespace(&namespace_ident, namespace_properties)
            .await
            .unwrap();

        let table_name = "tbl1";
        let expected_table_ident = TableIdent::new(namespace_ident.clone(), table_name.into());
        let expected_table_metadata_location_regex =
            format!("^{warehouse_location}/a/tbl1/metadata/00000-{UUID_REGEX_STR}.metadata.json$");

        let table = catalog
            .create_table(
                &namespace_ident,
                TableCreation::builder()
                    .name(table_name.into())
                    .schema(simple_table_schema())
                    // no location specified for table
                    .build(),
            )
            .await
            .unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);

        let table = catalog.load_table(&expected_table_ident).await.unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);
    }

    #[tokio::test]
    async fn test_create_table_in_nested_namespace_falls_back_to_warehouse_location_if_both_table_location_and_namespace_location_are_missing()
     {
        let warehouse_location = temp_path();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    warehouse_location.clone(),
                )]),
            )
            .await
            .unwrap();

        let namespace_ident = NamespaceIdent::new("a".into());
        catalog
            // note: no location specified in namespace_properties
            .create_namespace(&namespace_ident, HashMap::new())
            .await
            .unwrap();

        let nested_namespace_ident = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        catalog
            // note: no location specified in namespace_properties
            .create_namespace(&nested_namespace_ident, HashMap::new())
            .await
            .unwrap();

        let table_name = "tbl1";
        let expected_table_ident =
            TableIdent::new(nested_namespace_ident.clone(), table_name.into());
        let expected_table_metadata_location_regex = format!(
            "^{warehouse_location}/a/b/tbl1/metadata/00000-{UUID_REGEX_STR}.metadata.json$"
        );

        let table = catalog
            .create_table(
                &nested_namespace_ident,
                TableCreation::builder()
                    .name(table_name.into())
                    .schema(simple_table_schema())
                    // no location specified for table
                    .build(),
            )
            .await
            .unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);

        let table = catalog.load_table(&expected_table_ident).await.unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);
    }

    #[tokio::test]
    async fn test_create_table_throws_error_if_table_location_and_namespace_location_and_warehouse_location_are_missing()
     {
        let catalog = MemoryCatalogBuilder::default()
            .load("memory", HashMap::from([]))
            .await;

        assert!(catalog.is_err());
        assert_eq!(
            catalog.unwrap_err().to_string(),
            "DataInvalid => Catalog warehouse is required"
        );
    }

    #[tokio::test]
    async fn test_create_table_throws_error_if_table_with_same_name_already_exists() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;
        let table_name = "tbl1";
        let table_ident = TableIdent::new(namespace_ident.clone(), table_name.into());
        create_table(&catalog, &table_ident).await;

        let tmp_dir = TempDir::new().unwrap();
        let location = tmp_dir.path().to_str().unwrap().to_string();

        assert_eq!(
            catalog
                .create_table(
                    &namespace_ident,
                    TableCreation::builder()
                        .name(table_name.into())
                        .schema(simple_table_schema())
                        .location(location)
                        .build()
                )
                .await
                .unwrap_err()
                .to_string(),
            format!(
                "TableAlreadyExists => Cannot create table {:?}. Table already exists.",
                &table_ident
            )
        );
    }

    #[tokio::test]
    async fn test_list_tables_returns_empty_vector() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;

        assert_eq!(catalog.list_tables(&namespace_ident).await.unwrap(), vec![]);
    }

    #[tokio::test]
    async fn test_list_tables_returns_a_single_table() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        create_namespace(&catalog, &namespace_ident).await;

        let table_ident = TableIdent::new(namespace_ident.clone(), "tbl1".into());
        create_table(&catalog, &table_ident).await;

        assert_eq!(catalog.list_tables(&namespace_ident).await.unwrap(), vec![
            table_ident
        ]);
    }

    #[tokio::test]
    async fn test_list_tables_returns_multiple_tables() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        create_namespace(&catalog, &namespace_ident).await;

        let table_ident_1 = TableIdent::new(namespace_ident.clone(), "tbl1".into());
        let table_ident_2 = TableIdent::new(namespace_ident.clone(), "tbl2".into());
        let _ = create_tables(&catalog, vec![&table_ident_1, &table_ident_2]).await;

        assert_eq!(
            to_set(catalog.list_tables(&namespace_ident).await.unwrap()),
            to_set(vec![table_ident_1, table_ident_2])
        );
    }

    #[tokio::test]
    async fn test_list_tables_returns_tables_from_correct_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_1 = NamespaceIdent::new("n1".into());
        let namespace_ident_2 = NamespaceIdent::new("n2".into());
        create_namespaces(&catalog, &vec![&namespace_ident_1, &namespace_ident_2]).await;

        let table_ident_1 = TableIdent::new(namespace_ident_1.clone(), "tbl1".into());
        let table_ident_2 = TableIdent::new(namespace_ident_1.clone(), "tbl2".into());
        let table_ident_3 = TableIdent::new(namespace_ident_2.clone(), "tbl1".into());
        let _ = create_tables(&catalog, vec![
            &table_ident_1,
            &table_ident_2,
            &table_ident_3,
        ])
        .await;

        assert_eq!(
            to_set(catalog.list_tables(&namespace_ident_1).await.unwrap()),
            to_set(vec![table_ident_1, table_ident_2])
        );

        assert_eq!(
            to_set(catalog.list_tables(&namespace_ident_2).await.unwrap()),
            to_set(vec![table_ident_3])
        );
    }

    #[tokio::test]
    async fn test_list_tables_returns_table_under_nested_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespaces(&catalog, &vec![&namespace_ident_a, &namespace_ident_a_b]).await;

        let table_ident = TableIdent::new(namespace_ident_a_b.clone(), "tbl1".into());
        create_table(&catalog, &table_ident).await;

        assert_eq!(
            catalog.list_tables(&namespace_ident_a_b).await.unwrap(),
            vec![table_ident]
        );
    }

    #[tokio::test]
    async fn test_list_tables_throws_error_if_namespace_doesnt_exist() {
        let catalog = new_memory_catalog().await;

        let non_existent_namespace_ident = NamespaceIdent::new("n1".into());

        assert_eq!(
            catalog
                .list_tables(&non_existent_namespace_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("NamespaceNotFound => No such namespace: {non_existent_namespace_ident:?}"),
        );
    }

    #[tokio::test]
    async fn test_drop_table() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        create_namespace(&catalog, &namespace_ident).await;
        let table_ident = TableIdent::new(namespace_ident.clone(), "tbl1".into());
        create_table(&catalog, &table_ident).await;

        catalog.drop_table(&table_ident).await.unwrap();
    }

    #[tokio::test]
    async fn test_drop_table_drops_table_under_nested_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespaces(&catalog, &vec![&namespace_ident_a, &namespace_ident_a_b]).await;

        let table_ident = TableIdent::new(namespace_ident_a_b.clone(), "tbl1".into());
        create_table(&catalog, &table_ident).await;

        catalog.drop_table(&table_ident).await.unwrap();

        assert_eq!(
            catalog.list_tables(&namespace_ident_a_b).await.unwrap(),
            vec![]
        );
    }

    #[tokio::test]
    async fn test_drop_table_throws_error_if_namespace_doesnt_exist() {
        let catalog = new_memory_catalog().await;

        let non_existent_namespace_ident = NamespaceIdent::new("n1".into());
        let non_existent_table_ident =
            TableIdent::new(non_existent_namespace_ident.clone(), "tbl1".into());

        assert_eq!(
            catalog
                .drop_table(&non_existent_table_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("NamespaceNotFound => No such namespace: {non_existent_namespace_ident:?}"),
        );
    }

    #[tokio::test]
    async fn test_drop_table_throws_error_if_table_doesnt_exist() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        create_namespace(&catalog, &namespace_ident).await;

        let non_existent_table_ident = TableIdent::new(namespace_ident.clone(), "tbl1".into());

        assert_eq!(
            catalog
                .drop_table(&non_existent_table_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("TableNotFound => No such table: {non_existent_table_ident:?}"),
        );
    }

    #[tokio::test]
    async fn test_table_exists_returns_true() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        create_namespace(&catalog, &namespace_ident).await;
        let table_ident = TableIdent::new(namespace_ident.clone(), "tbl1".into());
        create_table(&catalog, &table_ident).await;

        assert!(catalog.table_exists(&table_ident).await.unwrap());
    }

    #[tokio::test]
    async fn test_table_exists_returns_false() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        create_namespace(&catalog, &namespace_ident).await;
        let non_existent_table_ident = TableIdent::new(namespace_ident.clone(), "tbl1".into());

        assert!(
            !catalog
                .table_exists(&non_existent_table_ident)
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_table_exists_under_nested_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespaces(&catalog, &vec![&namespace_ident_a, &namespace_ident_a_b]).await;

        let table_ident = TableIdent::new(namespace_ident_a_b.clone(), "tbl1".into());
        create_table(&catalog, &table_ident).await;

        assert!(catalog.table_exists(&table_ident).await.unwrap());

        let non_existent_table_ident = TableIdent::new(namespace_ident_a_b.clone(), "tbl2".into());
        assert!(
            !catalog
                .table_exists(&non_existent_table_ident)
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_table_exists_throws_error_if_namespace_doesnt_exist() {
        let catalog = new_memory_catalog().await;

        let non_existent_namespace_ident = NamespaceIdent::new("n1".into());
        let non_existent_table_ident =
            TableIdent::new(non_existent_namespace_ident.clone(), "tbl1".into());

        assert_eq!(
            catalog
                .table_exists(&non_existent_table_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("NamespaceNotFound => No such namespace: {non_existent_namespace_ident:?}"),
        );
    }

    #[tokio::test]
    async fn test_rename_table_in_same_namespace() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        create_namespace(&catalog, &namespace_ident).await;
        let src_table_ident = TableIdent::new(namespace_ident.clone(), "tbl1".into());
        let dst_table_ident = TableIdent::new(namespace_ident.clone(), "tbl2".into());
        create_table(&catalog, &src_table_ident).await;

        catalog
            .rename_table(&src_table_ident, &dst_table_ident)
            .await
            .unwrap();

        assert_eq!(catalog.list_tables(&namespace_ident).await.unwrap(), vec![
            dst_table_ident
        ],);
    }

    #[tokio::test]
    async fn test_rename_table_across_namespaces() {
        let catalog = new_memory_catalog().await;
        let src_namespace_ident = NamespaceIdent::new("a".into());
        let dst_namespace_ident = NamespaceIdent::new("b".into());
        create_namespaces(&catalog, &vec![&src_namespace_ident, &dst_namespace_ident]).await;
        let src_table_ident = TableIdent::new(src_namespace_ident.clone(), "tbl1".into());
        let dst_table_ident = TableIdent::new(dst_namespace_ident.clone(), "tbl2".into());
        create_table(&catalog, &src_table_ident).await;

        catalog
            .rename_table(&src_table_ident, &dst_table_ident)
            .await
            .unwrap();

        assert_eq!(
            catalog.list_tables(&src_namespace_ident).await.unwrap(),
            vec![],
        );

        assert_eq!(
            catalog.list_tables(&dst_namespace_ident).await.unwrap(),
            vec![dst_table_ident],
        );
    }

    #[tokio::test]
    async fn test_rename_table_src_table_is_same_as_dst_table() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        create_namespace(&catalog, &namespace_ident).await;
        let table_ident = TableIdent::new(namespace_ident.clone(), "tbl".into());
        create_table(&catalog, &table_ident).await;

        catalog
            .rename_table(&table_ident, &table_ident)
            .await
            .unwrap();

        assert_eq!(catalog.list_tables(&namespace_ident).await.unwrap(), vec![
            table_ident
        ],);
    }

    #[tokio::test]
    async fn test_rename_table_across_nested_namespaces() {
        let catalog = new_memory_catalog().await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_a_b_c = NamespaceIdent::from_strs(vec!["a", "b", "c"]).unwrap();
        create_namespaces(&catalog, &vec![
            &namespace_ident_a,
            &namespace_ident_a_b,
            &namespace_ident_a_b_c,
        ])
        .await;

        let src_table_ident = TableIdent::new(namespace_ident_a_b_c.clone(), "tbl1".into());
        create_tables(&catalog, vec![&src_table_ident]).await;

        let dst_table_ident = TableIdent::new(namespace_ident_a_b.clone(), "tbl1".into());
        catalog
            .rename_table(&src_table_ident, &dst_table_ident)
            .await
            .unwrap();

        assert!(!catalog.table_exists(&src_table_ident).await.unwrap());

        assert!(catalog.table_exists(&dst_table_ident).await.unwrap());
    }

    #[tokio::test]
    async fn test_rename_table_throws_error_if_src_namespace_doesnt_exist() {
        let catalog = new_memory_catalog().await;

        let non_existent_src_namespace_ident = NamespaceIdent::new("n1".into());
        let src_table_ident =
            TableIdent::new(non_existent_src_namespace_ident.clone(), "tbl1".into());

        let dst_namespace_ident = NamespaceIdent::new("n2".into());
        create_namespace(&catalog, &dst_namespace_ident).await;
        let dst_table_ident = TableIdent::new(dst_namespace_ident.clone(), "tbl1".into());

        assert_eq!(
            catalog
                .rename_table(&src_table_ident, &dst_table_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("NamespaceNotFound => No such namespace: {non_existent_src_namespace_ident:?}"),
        );
    }

    #[tokio::test]
    async fn test_rename_table_throws_error_if_dst_namespace_doesnt_exist() {
        let catalog = new_memory_catalog().await;
        let src_namespace_ident = NamespaceIdent::new("n1".into());
        let src_table_ident = TableIdent::new(src_namespace_ident.clone(), "tbl1".into());
        create_namespace(&catalog, &src_namespace_ident).await;
        create_table(&catalog, &src_table_ident).await;

        let non_existent_dst_namespace_ident = NamespaceIdent::new("n2".into());
        let dst_table_ident =
            TableIdent::new(non_existent_dst_namespace_ident.clone(), "tbl1".into());
        assert_eq!(
            catalog
                .rename_table(&src_table_ident, &dst_table_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("NamespaceNotFound => No such namespace: {non_existent_dst_namespace_ident:?}"),
        );
    }

    #[tokio::test]
    async fn test_rename_table_throws_error_if_src_table_doesnt_exist() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        create_namespace(&catalog, &namespace_ident).await;
        let src_table_ident = TableIdent::new(namespace_ident.clone(), "tbl1".into());
        let dst_table_ident = TableIdent::new(namespace_ident.clone(), "tbl2".into());

        assert_eq!(
            catalog
                .rename_table(&src_table_ident, &dst_table_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("TableNotFound => No such table: {src_table_ident:?}"),
        );
    }

    #[tokio::test]
    async fn test_rename_table_throws_error_if_dst_table_already_exists() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        create_namespace(&catalog, &namespace_ident).await;
        let src_table_ident = TableIdent::new(namespace_ident.clone(), "tbl1".into());
        let dst_table_ident = TableIdent::new(namespace_ident.clone(), "tbl2".into());
        create_tables(&catalog, vec![&src_table_ident, &dst_table_ident]).await;

        assert_eq!(
            catalog
                .rename_table(&src_table_ident, &dst_table_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!(
                "TableAlreadyExists => Cannot create table {:?}. Table already exists.",
                &dst_table_ident
            ),
        );
    }

    #[tokio::test]
    async fn test_register_table() {
        // Create a catalog and namespace
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("test_namespace".into());
        create_namespace(&catalog, &namespace_ident).await;

        // Create a table to get a valid metadata file
        let source_table_ident = TableIdent::new(namespace_ident.clone(), "source_table".into());
        create_table(&catalog, &source_table_ident).await;

        // Get the metadata location from the source table
        let source_table = catalog.load_table(&source_table_ident).await.unwrap();
        let metadata_location = source_table.metadata_location().unwrap().to_string();

        // Register a new table using the same metadata location
        let register_table_ident =
            TableIdent::new(namespace_ident.clone(), "register_table".into());
        let registered_table = catalog
            .register_table(&register_table_ident, metadata_location.clone())
            .await
            .unwrap();

        // Verify the registered table has the correct identifier
        assert_eq!(registered_table.identifier(), &register_table_ident);

        // Verify the registered table has the correct metadata location
        assert_eq!(
            registered_table.metadata_location().unwrap().to_string(),
            metadata_location
        );

        // Verify the table exists in the catalog
        assert!(catalog.table_exists(&register_table_ident).await.unwrap());

        // Verify we can load the registered table
        let loaded_table = catalog.load_table(&register_table_ident).await.unwrap();
        assert_eq!(loaded_table.identifier(), &register_table_ident);
        assert_eq!(
            loaded_table.metadata_location().unwrap().to_string(),
            metadata_location
        );
    }

    #[tokio::test]
    async fn test_update_table() {
        let catalog = new_memory_catalog().await;

        let table = create_table_with_namespace(&catalog).await;

        // Assert the table doesn't contain the update yet
        assert!(!table.metadata().properties().contains_key("key"));

        // Update table metadata
        let tx = Transaction::new(&table);
        let updated_table = tx
            .update_table_properties()
            .set("key".to_string(), "value".to_string())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        assert_eq!(
            updated_table.metadata().properties().get("key").unwrap(),
            "value"
        );

        assert_eq!(table.identifier(), updated_table.identifier());
        assert_eq!(table.metadata().uuid(), updated_table.metadata().uuid());
        // `last_updated_ms` is millisecond-precision wall-clock; a fast update can land in the same
        // millisecond as table creation, so this must be `<=` (strict `<` is flaky under parallel
        // test load). That an update actually occurred is asserted via the metadata-log growth below.
        assert!(table.metadata().last_updated_ms() <= updated_table.metadata().last_updated_ms());
        assert_ne!(table.metadata_location(), updated_table.metadata_location());

        assert!(
            table.metadata().metadata_log().len() < updated_table.metadata().metadata_log().len()
        );
    }

    #[tokio::test]
    async fn test_update_table_fails_if_table_doesnt_exist() {
        let catalog = new_memory_catalog().await;

        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;

        // This table is not known to the catalog.
        let table_ident = TableIdent::new(namespace_ident, "test".to_string());
        let table = build_table(table_ident);

        let tx = Transaction::new(&table);
        let err = tx
            .update_table_properties()
            .set("key".to_string(), "value".to_string())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::TableNotFound);
    }

    fn build_table(ident: TableIdent) -> Table {
        let file_io = FileIO::new_with_fs();

        let temp_dir = TempDir::new().unwrap();
        let location = temp_dir.path().to_str().unwrap().to_string();

        let table_creation = TableCreation::builder()
            .name(ident.name().to_string())
            .schema(simple_table_schema())
            .location(location)
            .build();
        let metadata = TableMetadataBuilder::from_table_creation(table_creation)
            .unwrap()
            .build()
            .unwrap()
            .metadata;

        Table::builder()
            .identifier(ident)
            .metadata(metadata)
            .file_io(file_io)
            .build()
            .unwrap()
    }

    // ========================================================================
    // View CRUD lifecycle tests (the MemoryCatalog view surface).
    // ========================================================================

    fn simple_view_schema() -> Schema {
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(1, "event_count", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap()
    }

    fn sql_representations(sql: &str) -> crate::spec::ViewRepresentations {
        crate::spec::ViewRepresentations(vec![crate::spec::ViewRepresentation::Sql(
            crate::spec::SqlViewRepresentation {
                sql: sql.to_string(),
                dialect: "spark".to_string(),
            },
        )])
    }

    async fn create_view<C: Catalog>(catalog: &C, view_ident: &TableIdent, sql: &str) -> View {
        let location = format!("{}/{}", temp_path(), view_ident.name());
        catalog
            .create_view(
                &view_ident.namespace,
                ViewCreation::builder()
                    .name(view_ident.name().to_string())
                    .location(location)
                    .schema(simple_view_schema())
                    .default_namespace(view_ident.namespace.clone())
                    .representations(sql_representations(sql))
                    .build(),
            )
            .await
            .unwrap()
    }

    // RISK: a fully separate view namespace — creating/loading/listing views must NOT collide with
    // tables of the same name and must round-trip the view metadata through FileIO.
    #[tokio::test]
    async fn test_view_create_load_and_list() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("ns".into());
        create_namespace(&catalog, &namespace_ident).await;
        let view_ident = TableIdent::new(namespace_ident.clone(), "v1".into());

        let view = create_view(&catalog, &view_ident, "SELECT 1 AS event_count").await;
        assert_eq!(view.identifier(), &view_ident);
        assert_eq!(view.metadata().current_version_id(), 1);

        assert!(catalog.view_exists(&view_ident).await.unwrap());
        let loaded = catalog.load_view(&view_ident).await.unwrap();
        assert_eq!(loaded.metadata().uuid(), view.metadata().uuid());
        assert_eq!(loaded.metadata().versions().count(), 1);

        let views = catalog.list_views(&namespace_ident).await.unwrap();
        assert_eq!(views, vec![view_ident]);
    }

    // RISK: creating a view that already exists must fail loudly (Java AlreadyExistsException).
    #[tokio::test]
    async fn test_view_create_duplicate_fails() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("ns".into());
        create_namespace(&catalog, &namespace_ident).await;
        let view_ident = TableIdent::new(namespace_ident, "v1".into());

        create_view(&catalog, &view_ident, "SELECT 1 AS event_count").await;
        let location = format!("{}/{}", temp_path(), "v1");
        let error = catalog
            .create_view(
                &view_ident.namespace,
                ViewCreation::builder()
                    .name("v1".to_string())
                    .location(location)
                    .schema(simple_view_schema())
                    .default_namespace(view_ident.namespace.clone())
                    .representations(sql_representations("SELECT 1"))
                    .build(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ViewAlreadyExists);
    }

    // RISK: the full create→update→load→rename→drop lifecycle — a replace must flip the current
    // version, append the version log, and KEEP the old version intact; rename must move the
    // pointer; drop must remove it. This is the load-bearing catalog-CRUD e2e.
    #[tokio::test]
    async fn test_view_full_lifecycle_replace_rename_drop() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("ns".into());
        create_namespace(&catalog, &namespace_ident).await;
        let view_ident = TableIdent::new(namespace_ident.clone(), "v1".into());

        let view = create_view(&catalog, &view_ident, "SELECT 1 AS event_count").await;
        let original_uuid = view.metadata().uuid();
        let original_location = view.metadata_location().unwrap().to_string();

        // Replace the version with a genuinely different query.
        let commit = view
            .replace_version()
            .with_query("spark", "SELECT 2 AS event_count")
            .with_schema(simple_view_schema())
            .with_default_namespace(namespace_ident.clone())
            .to_commit()
            .unwrap();
        let updated = catalog.update_view(commit).await.unwrap();

        // The metadata-file pointer advanced.
        assert_ne!(updated.metadata_location().unwrap(), original_location);
        // UUID is preserved across the replace.
        assert_eq!(updated.metadata().uuid(), original_uuid);

        // Loading shows the NEW current version, the appended log, and the OLD version intact.
        let loaded = catalog.load_view(&view_ident).await.unwrap();
        assert_eq!(loaded.metadata().current_version_id(), 2);
        assert_eq!(loaded.metadata().versions().count(), 2);
        assert!(loaded.metadata().version_by_id(1).is_some());
        assert_eq!(
            loaded
                .metadata()
                .history()
                .iter()
                .map(|entry| entry.version_id())
                .collect::<Vec<_>>(),
            vec![1, 2]
        );

        // Rename the view, then confirm the source is gone and the destination loads.
        let renamed_ident = TableIdent::new(namespace_ident.clone(), "v2".into());
        catalog
            .rename_view(&view_ident, &renamed_ident)
            .await
            .unwrap();
        assert!(!catalog.view_exists(&view_ident).await.unwrap());
        assert!(catalog.view_exists(&renamed_ident).await.unwrap());
        let renamed = catalog.load_view(&renamed_ident).await.unwrap();
        assert_eq!(renamed.metadata().uuid(), original_uuid);
        assert_eq!(renamed.metadata().current_version_id(), 2);

        // Drop the view; it is then gone and a re-load fails.
        catalog.drop_view(&renamed_ident).await.unwrap();
        assert!(!catalog.view_exists(&renamed_ident).await.unwrap());
        let error = catalog.load_view(&renamed_ident).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ViewNotFound);
    }

    // RISK: an update_properties commit must persist properties through the catalog round-trip.
    #[tokio::test]
    async fn test_view_update_properties_through_catalog() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("ns".into());
        create_namespace(&catalog, &namespace_ident).await;
        let view_ident = TableIdent::new(namespace_ident, "v1".into());

        let view = create_view(&catalog, &view_ident, "SELECT 1 AS event_count").await;
        let commit = view
            .update_properties()
            .set("comment", "daily counts")
            .unwrap()
            .to_commit()
            .unwrap();
        catalog.update_view(commit).await.unwrap();

        let loaded = catalog.load_view(&view_ident).await.unwrap();
        assert_eq!(
            loaded.metadata().properties().get("comment"),
            Some(&"daily counts".to_string())
        );
    }

    // RISK: committing the SAME version twice through the catalog must REUSE the existing version —
    // Java reuses rather than minting a new id, so the version count stays constant.
    #[tokio::test]
    async fn test_view_replace_identical_version_reuses() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("ns".into());
        create_namespace(&catalog, &namespace_ident).await;
        let view_ident = TableIdent::new(namespace_ident.clone(), "v1".into());

        let view = create_view(&catalog, &view_ident, "SELECT 1 AS event_count").await;
        // Replace with the IDENTICAL representation already at version 1.
        let commit = view
            .replace_version()
            .with_query("spark", "SELECT 1 AS event_count")
            .with_schema(simple_view_schema())
            .with_default_namespace(namespace_ident)
            .to_commit()
            .unwrap();
        catalog.update_view(commit).await.unwrap();

        let loaded = catalog.load_view(&view_ident).await.unwrap();
        assert_eq!(loaded.metadata().versions().count(), 1);
        assert_eq!(loaded.metadata().current_version_id(), 1);
    }

    // RISK: a catalog without view support inherits the default trait methods that error rather
    // than silently no-op — pin that a load on a non-existent view errors (ViewNotFound), proving
    // the MemoryCatalog override is wired (not the FeatureUnsupported default).
    #[tokio::test]
    async fn test_view_load_missing_errors_view_not_found() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("ns".into());
        create_namespace(&catalog, &namespace_ident).await;
        let view_ident = TableIdent::new(namespace_ident, "absent".into());

        let error = catalog.load_view(&view_ident).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ViewNotFound);
        assert!(!catalog.view_exists(&view_ident).await.unwrap());
    }

    // RISK: tables and views share ONE name space in a catalog (Java InMemoryCatalog). Creating a
    // view where a TABLE already exists — or a table where a VIEW exists — must be REJECTED, not
    // silently coexist. A shadowing pair lets a later `loadTable`/`loadView` resolve the wrong kind.
    #[tokio::test]
    async fn test_view_and_table_name_collision_rejected_both_directions() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("ns".into());
        create_namespace(&catalog, &namespace_ident).await;

        // A TABLE exists; creating a VIEW with the same name is rejected ("Table with same name").
        let table_ident = TableIdent::new(namespace_ident.clone(), "shared_a".into());
        create_table(&catalog, &table_ident).await;
        let view_over_table = catalog
            .create_view(
                &namespace_ident,
                ViewCreation::builder()
                    .name("shared_a".to_string())
                    .location(format!("{}/shared_a_view", temp_path()))
                    .schema(simple_view_schema())
                    .default_namespace(namespace_ident.clone())
                    .representations(sql_representations("SELECT 1"))
                    .build(),
            )
            .await
            .unwrap_err();
        assert_eq!(view_over_table.kind(), ErrorKind::TableAlreadyExists);
        // The view was NOT created.
        assert!(!catalog.view_exists(&table_ident).await.unwrap());

        // A VIEW exists; creating a TABLE with the same name is rejected ("View with same name").
        let view_ident = TableIdent::new(namespace_ident.clone(), "shared_b".into());
        create_view(&catalog, &view_ident, "SELECT 1 AS event_count").await;
        let table_over_view = catalog
            .create_table(
                &namespace_ident,
                TableCreation::builder()
                    .name("shared_b".to_string())
                    .location(format!("{}/shared_b", temp_path()))
                    .schema(simple_view_schema())
                    .build(),
            )
            .await
            .unwrap_err();
        assert_eq!(table_over_view.kind(), ErrorKind::ViewAlreadyExists);
        assert!(!catalog.table_exists(&view_ident).await.unwrap());
    }

    // RISK: a rename must also respect the shared name space — renaming a view onto a name a TABLE
    // already holds must be rejected (Java InMemoryCatalog.renameView checks `tables.containsKey`).
    #[tokio::test]
    async fn test_rename_view_onto_existing_table_rejected() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("ns".into());
        create_namespace(&catalog, &namespace_ident).await;

        let view_ident = TableIdent::new(namespace_ident.clone(), "v_src".into());
        create_view(&catalog, &view_ident, "SELECT 1 AS event_count").await;
        let table_ident = TableIdent::new(namespace_ident.clone(), "t_dst".into());
        create_table(&catalog, &table_ident).await;

        let error = catalog
            .rename_view(&view_ident, &table_ident)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::TableAlreadyExists);
        // The source view is untouched and the table is intact.
        assert!(catalog.view_exists(&view_ident).await.unwrap());
        assert!(catalog.table_exists(&table_ident).await.unwrap());
    }

    // ========================================================================
    // Optimistic-concurrency parity (increment O1) — the location-CAS in MemoryCatalog
    // `update_table` / `update_view`, mirroring Java `InMemory{Table,View}Operations.doCommit`.
    // ========================================================================

    // RISK: two REPLACE commits built from the SAME base view, applied sequentially, must NOT
    // last-write-win — the second is STALE and Java `InMemoryViewOperations.doCommit` rejects it
    // with `CommitFailedException`. The `[AssertViewUUID]` requirement is INVARIANT across replaces,
    // so the location-CAS is the ONLY thing that can detect this (the bug the U1 reviewer pinned:
    // before the CAS, the second commit silently advanced versions 1→2→3, clobbering the winner).
    #[tokio::test]
    async fn test_view_stale_second_replace_conflicts_via_location_cas() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("ns".into());
        create_namespace(&catalog, &namespace_ident).await;
        let view_ident = TableIdent::new(namespace_ident.clone(), "v1".into());

        // Both commits are built from the SAME base `view` handle (same metadata location).
        let view = create_view(&catalog, &view_ident, "SELECT 1 AS event_count").await;
        let base_location = view.metadata_location().unwrap().to_string();

        let first_commit = view
            .replace_version()
            .with_query("spark", "SELECT 2 AS event_count")
            .with_schema(simple_view_schema())
            .with_default_namespace(namespace_ident.clone())
            .to_commit()
            .unwrap();
        let second_commit = view
            .replace_version()
            .with_query("spark", "SELECT 3 AS event_count")
            .with_schema(simple_view_schema())
            .with_default_namespace(namespace_ident.clone())
            .to_commit()
            .unwrap();

        // Both commits carry the base location they were built against.
        assert_eq!(
            first_commit.base_metadata_location(),
            Some(base_location.as_str())
        );
        assert_eq!(
            second_commit.base_metadata_location(),
            Some(base_location.as_str())
        );

        // The first commit lands: the stored location still equals the base, so the CAS passes.
        let winner = catalog.update_view(first_commit).await.unwrap();
        assert_eq!(winner.metadata().current_version_id(), 2);

        // The second commit is now STALE (stored location advanced past its base) and must be
        // rejected with a retryable commit conflict, NOT silently applied.
        let error = catalog.update_view(second_commit).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::CatalogCommitConflicts);
        assert!(error.retryable());
        assert!(
            error
                .to_string()
                .contains("because it has been concurrently modified to")
        );

        // The store still reflects the WINNER — the loser did not overwrite it.
        let loaded = catalog.load_view(&view_ident).await.unwrap();
        assert_eq!(loaded.metadata().current_version_id(), 2);
        assert_eq!(loaded.metadata_location(), winner.metadata_location());
    }

    // RISK: the same stale-commit shape for TABLES, through a property-only `TableCommit`. A
    // property update carries an EMPTY requirement set (`UpdatePropertiesAction` emits no
    // requirements), so NO `TableRequirement` can catch the staleness — only the location-CAS can.
    // This is the case the prompt asks to pin: "ONLY the location CAS can fire".
    #[tokio::test]
    async fn test_table_stale_property_commit_conflicts_only_location_cas_can_fire() {
        let catalog = new_memory_catalog().await;
        let table = create_table_with_namespace(&catalog).await;
        let base_location = table.metadata_location().unwrap().to_string();

        // Two property-only commits from the SAME base location. They carry NO requirements, so the
        // CAS is the sole guard. Build them directly so the transaction's reload cannot mask the race.
        let first_commit = TableCommit::builder()
            .ident(table.identifier().clone())
            .requirements(vec![])
            .updates(vec![TableUpdate::SetProperties {
                updates: HashMap::from([("round".to_string(), "first".to_string())]),
            }])
            .base_metadata_location(Some(base_location.clone()))
            .build();
        let second_commit = TableCommit::builder()
            .ident(table.identifier().clone())
            .requirements(vec![])
            .updates(vec![TableUpdate::SetProperties {
                updates: HashMap::from([("round".to_string(), "second".to_string())]),
            }])
            .base_metadata_location(Some(base_location.clone()))
            .build();

        // First lands; the stored location matched the base.
        let winner = catalog.update_table(first_commit).await.unwrap();
        assert_eq!(
            winner.metadata().properties().get("round").unwrap(),
            "first"
        );

        // Second is stale: the location advanced, and with no requirements ONLY the CAS fires.
        let error = catalog.update_table(second_commit).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::CatalogCommitConflicts);
        assert!(error.retryable());
        assert!(
            error
                .to_string()
                .contains("because it has been concurrently modified to")
        );

        // The winner's property survived — the loser did not overwrite the pointer.
        let loaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            loaded.metadata().properties().get("round").unwrap(),
            "first"
        );
        assert_eq!(loaded.metadata_location(), winner.metadata_location());
    }

    // RISK: the happy path must be UNCHANGED — a single non-stale property commit (built from the
    // current base) still lands. A CAS that rejected non-stale commits would break every writer.
    #[tokio::test]
    async fn test_table_non_stale_commit_still_succeeds_with_cas() {
        let catalog = new_memory_catalog().await;
        let table = create_table_with_namespace(&catalog).await;

        let commit = TableCommit::builder()
            .ident(table.identifier().clone())
            .requirements(vec![])
            .updates(vec![TableUpdate::SetProperties {
                updates: HashMap::from([("key".to_string(), "value".to_string())]),
            }])
            .base_metadata_location(table.metadata_location().map(str::to_string))
            .build();

        let updated = catalog.update_table(commit).await.unwrap();
        assert_eq!(updated.metadata().properties().get("key").unwrap(), "value");
        assert_ne!(updated.metadata_location(), table.metadata_location());
    }

    // RISK: a stale transaction must RECOVER end-to-end. Two `Transaction`s start from the same base
    // table; the first commits, then the second commits. The transaction's `do_commit` reloads the
    // (now-advanced) base before building its `TableCommit`, so the location-CAS passes against the
    // refreshed base and the second commit lands on TOP of the winner — refresh-and-retry resilience
    // through the real commit machinery, not a silent last-write-win. Both properties survive.
    #[tokio::test]
    async fn test_table_two_transactions_from_same_base_both_land_via_refresh() {
        let catalog = new_memory_catalog().await;
        let table = create_table_with_namespace(&catalog).await;

        // First transaction commits a property.
        let tx_first = Transaction::new(&table);
        let after_first = tx_first
            .update_table_properties()
            .set("first".to_string(), "1".to_string())
            .apply(tx_first)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
        assert_eq!(
            after_first.metadata().properties().get("first").unwrap(),
            "1"
        );

        // Second transaction is built from the STALE original `table` handle, but its `do_commit`
        // refreshes to the winner's base before committing — so it succeeds and the first property
        // is preserved (the refresh re-applied on top of the winner, no clobber).
        let tx_second = Transaction::new(&table);
        let after_second = tx_second
            .update_table_properties()
            .set("second".to_string(), "2".to_string())
            .apply(tx_second)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
        assert_eq!(
            after_second.metadata().properties().get("second").unwrap(),
            "2"
        );

        // Both writes survive end-to-end — the second did not last-write-win over the first.
        let loaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(loaded.metadata().properties().get("first").unwrap(), "1");
        assert_eq!(loaded.metadata().properties().get("second").unwrap(), "2");
    }

    #[tokio::test]
    async fn test_name_returns_configured_name() {
        let catalog = new_memory_catalog().await;
        // `new_memory_catalog` loads with the name "memory"; the MemoryCatalog::new fix must
        // retain it (it previously dropped name+props). An accessor that returned the sentinel
        // or an empty string would fail here.
        assert_eq!(catalog.name(), "memory");
        assert_ne!(catalog.name(), UNNAMED_CATALOG);
    }

    #[tokio::test]
    async fn test_properties_returns_configured_props() {
        // Build a catalog with both the required warehouse and an extra property; the
        // accessor must surface the extra property (the warehouse key is consumed and not a
        // retained config property).
        let warehouse_location = temp_path();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory_with_props",
                HashMap::from([
                    (MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_location),
                    ("k1".to_string(), "v1".to_string()),
                    ("k2".to_string(), "v2".to_string()),
                ]),
            )
            .await
            .unwrap();

        assert_eq!(catalog.name(), "memory_with_props");
        let props = catalog.properties();
        // Mutation guard: an accessor returning the empty-map default fails both of these.
        assert_eq!(props.get("k1").map(String::as_str), Some("v1"));
        assert_eq!(props.get("k2").map(String::as_str), Some("v2"));
        // The warehouse key is filtered out of the retained props (see `load`).
        assert!(!props.contains_key(MEMORY_CATALOG_WAREHOUSE));
    }

    #[tokio::test]
    async fn test_invalidate_table_default_is_noop() {
        let catalog = new_memory_catalog().await;
        let ident = TableIdent::new(NamespaceIdent::new("ns".to_string()), "tbl".to_string());
        // The default no-op (inherited; MemoryCatalog holds no cache) must not error even for
        // an unknown table — it mirrors Java's empty-`return` default body.
        catalog
            .invalidate_table(&ident)
            .await
            .expect("invalidate_table default must be a no-op");
    }

    #[tokio::test]
    async fn test_invalidate_view_default_is_noop() {
        let catalog = new_memory_catalog().await;
        let ident = TableIdent::new(NamespaceIdent::new("ns".to_string()), "vw".to_string());
        catalog
            .invalidate_view(&ident)
            .await
            .expect("invalidate_view default must be a no-op");
    }

    // ========================================================================
    // FK3 / scout #13 — lock hygiene: I/O outside the catalog mutex; atomicity pins.
    // ========================================================================

    /// RISK: a `register_table` whose metadata path is UNREACHABLE must leave NO catalog pointer
    /// (half-create refused). Read-before-insert is the guarantee; FileIO now runs outside the
    /// lock, so a failing read must still not insert. MUTATION: insert before read turns
    /// `table_exists` true and this test RED.
    #[tokio::test]
    async fn test_register_table_unreachable_metadata_refuses_half_create() {
        let catalog = new_memory_catalog().await;
        let namespace_ident = NamespaceIdent::new("ns".into());
        create_namespace(&catalog, &namespace_ident).await;

        let table_ident = TableIdent::new(namespace_ident, "half_create".into());
        let missing = format!("{}/definitely-missing/v1.metadata.json", temp_path());

        let err = catalog
            .register_table(&table_ident, missing)
            .await
            .expect_err("unreachable metadata must fail the register");
        // Any FileIO / parse failure is acceptable; the pin is catalog state, not error kind.
        let _ = err;

        assert!(
            !catalog
                .table_exists(&table_ident)
                .await
                .expect("table_exists"),
            "failed register must leave no catalog pointer (half-create refused)"
        );
        // Retry / create of the same identifier must still be possible.
        create_table(&catalog, &table_ident).await;
        assert!(
            catalog
                .table_exists(&table_ident)
                .await
                .expect("table_exists after create"),
            "after a failed register, a fresh create of the same ident must succeed"
        );
    }

    /// RISK: two stale property commits from the same base still conflict when FileIO is outside
    /// the lock — the flip-time CAS is the sole atomicity seam. (Re-asserts the O1 pin under the
    /// short-critical-section shape.)
    #[tokio::test]
    async fn test_table_stale_commit_conflicts_with_io_outside_lock() {
        let catalog = new_memory_catalog().await;
        let table = create_table_with_namespace(&catalog).await;
        let base_location = table.metadata_location().unwrap().to_string();

        let first_commit = TableCommit::builder()
            .ident(table.identifier().clone())
            .requirements(vec![])
            .updates(vec![TableUpdate::SetProperties {
                updates: HashMap::from([("round".to_string(), "first".to_string())]),
            }])
            .base_metadata_location(Some(base_location.clone()))
            .build();
        let second_commit = TableCommit::builder()
            .ident(table.identifier().clone())
            .requirements(vec![])
            .updates(vec![TableUpdate::SetProperties {
                updates: HashMap::from([("round".to_string(), "second".to_string())]),
            }])
            .base_metadata_location(Some(base_location))
            .build();

        let winner = catalog.update_table(first_commit).await.unwrap();
        let error = catalog.update_table(second_commit).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::CatalogCommitConflicts);
        assert!(error.retryable());

        let loaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            loaded.metadata().properties().get("round").unwrap(),
            "first"
        );
        assert_eq!(loaded.metadata_location(), winner.metadata_location());
    }

    /// Latency / concurrency note (structural, not a microbench): concurrent `load_table` while
    /// another task runs `update_table` must both complete. With I/O outside the lock the load is
    /// not serialized behind the update's metadata write — only the short pointer snapshot/CAS
    /// sections contend. This pin asserts liveness + winner visibility, not a wall-time histogram.
    #[tokio::test]
    async fn test_concurrent_load_during_update_completes() {
        let warehouse_location = temp_path();
        let catalog = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_location)]),
                )
                .await
                .expect("build memory catalog"),
        );
        let table = create_table_with_namespace(catalog.as_ref()).await;
        let ident = table.identifier().clone();
        let base_location = table.metadata_location().unwrap().to_string();

        let catalog_update = catalog.clone();
        let ident_update = ident.clone();
        let update = tokio::spawn(async move {
            let commit = TableCommit::builder()
                .ident(ident_update)
                .requirements(vec![])
                .updates(vec![TableUpdate::SetProperties {
                    updates: HashMap::from([("concurrent".to_string(), "yes".to_string())]),
                }])
                .base_metadata_location(Some(base_location))
                .build();
            catalog_update.update_table(commit).await
        });

        let catalog_load = catalog.clone();
        let ident_load = ident.clone();
        let load = tokio::spawn(async move { catalog_load.load_table(&ident_load).await });

        let updated = update
            .await
            .expect("update join")
            .expect("update must succeed");
        let loaded = load.await.expect("load join").expect("load must succeed");

        // Either the pre- or post-update pointer is a valid load; the update must have landed.
        assert_eq!(
            updated.metadata().properties().get("concurrent").unwrap(),
            "yes"
        );
        let final_table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(
            final_table
                .metadata()
                .properties()
                .get("concurrent")
                .unwrap(),
            "yes"
        );
        // Loaded table is a coherent snapshot of some metadata location known to the catalog.
        assert!(
            loaded.metadata_location().is_some(),
            "concurrent load must return a table with a metadata location"
        );
    }

    // ========================================================================
    // FK4.1 / scout #7 — metadata-pointer cache (opt-in, default OFF).
    // ========================================================================

    async fn new_memory_catalog_with_cache(cache: Arc<TableMetadataCache>) -> MemoryCatalog {
        let warehouse_location = temp_path();
        MemoryCatalogBuilder::default()
            .with_table_metadata_cache(cache)
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_location)]),
            )
            .await
            .expect("build memory catalog with table metadata cache")
    }

    /// Two loads of an unchanged pointer with the opt-in cache: create seeds the cache, so both
    /// loads are hits and body_fetches stay 0. MUTATION: skip cache lookup on load → misses > 0
    /// / body_fetches > 0 turns this RED.
    #[tokio::test]
    async fn test_fk4_1_two_loads_unchanged_pointer_zero_body_fetch() {
        let cache = Arc::new(TableMetadataCache::new());
        let catalog = new_memory_catalog_with_cache(cache.clone()).await;
        let table = create_table_with_namespace(&catalog).await;
        let ident = table.identifier().clone();
        let pointer = table.metadata_location().expect("pointer").to_string();

        cache.reset_stats();
        let first = catalog.load_table(&ident).await.expect("load 1");
        let second = catalog.load_table(&ident).await.expect("load 2");

        let stats = cache.stats();
        assert_eq!(
            stats.body_fetches, 0,
            "unchanged pointer after create-seed must not body-GET on load"
        );
        assert_eq!(stats.hits, 2, "both loads must hit the pointer cache");
        assert_eq!(stats.misses, 0);
        assert_eq!(
            first.metadata_location().unwrap(),
            pointer.as_str(),
            "load must surface the same catalog pointer"
        );
        assert_eq!(second.metadata_location().unwrap(), pointer.as_str());
        // Shared Arc from the cache (create seeded a distinct Arc; loads share the cached one).
        assert!(
            std::sync::Arc::ptr_eq(&first.metadata_ref(), &second.metadata_ref()),
            "two loads must share the cached TableMetadata Arc"
        );
    }

    /// Default OFF: builder without `with_table_metadata_cache` never records cache traffic and
    /// still loads correctly. (No injected Arc ⇒ no global/thread-local fallback.)
    #[tokio::test]
    async fn test_fk4_1_default_off_loads_without_cache() {
        let catalog = new_memory_catalog().await;
        let table = create_table_with_namespace(&catalog).await;
        let loaded = catalog
            .load_table(table.identifier())
            .await
            .expect("load without cache");
        assert_eq!(
            loaded.metadata_location(),
            table.metadata_location(),
            "default-OFF path must still resolve the catalog pointer"
        );
    }

    /// Commit advances the metadata location → new key → miss on next load (fail closed: never
    /// serve the previous pointer's Arc under a new location). Create+update seed the new key so
    /// the load after update is a hit on the *new* pointer; a second load is also a hit.
    #[tokio::test]
    async fn test_fk4_1_pointer_change_on_update_is_new_key() {
        let cache = Arc::new(TableMetadataCache::new());
        let catalog = new_memory_catalog_with_cache(cache.clone()).await;
        let table = create_table_with_namespace(&catalog).await;
        let ident = table.identifier().clone();
        let base_location = table.metadata_location().unwrap().to_string();

        let commit = TableCommit::builder()
            .ident(ident.clone())
            .requirements(vec![])
            .updates(vec![TableUpdate::SetProperties {
                updates: HashMap::from([("fk4".to_string(), "1".to_string())]),
            }])
            .base_metadata_location(Some(base_location.clone()))
            .build();
        let updated = catalog.update_table(commit).await.expect("update");
        let new_location = updated.metadata_location().unwrap().to_string();
        assert_ne!(
            new_location, base_location,
            "commit must publish a new metadata pointer"
        );

        cache.reset_stats();
        let loaded = catalog.load_table(&ident).await.expect("load after update");
        assert_eq!(loaded.metadata_location().unwrap(), new_location.as_str());
        assert_eq!(
            loaded
                .metadata()
                .properties()
                .get("fk4")
                .map(String::as_str),
            Some("1")
        );
        // update_table seeded the new pointer; load is a hit (zero body fetch).
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().body_fetches, 0);
        assert_eq!(cache.stats().misses, 0);
    }

    /// `invalidate_table` drops the location entry so the next load body-fetches again.
    #[tokio::test]
    async fn test_fk4_1_invalidate_table_evicts_pointer_entry() {
        let cache = Arc::new(TableMetadataCache::new());
        let catalog = new_memory_catalog_with_cache(cache.clone()).await;
        let table = create_table_with_namespace(&catalog).await;
        let ident = table.identifier().clone();
        let pointer = table.metadata_location().unwrap().to_string();

        // Confirm a hit first.
        cache.reset_stats();
        let _ = catalog.load_table(&ident).await.expect("warm");
        assert_eq!(cache.stats().hits, 1);

        catalog.invalidate_table(&ident).await.expect("invalidate");
        assert!(
            cache.lookup(&pointer, None).is_none(),
            "invalidate_table must drop the location entry"
        );

        cache.reset_stats();
        let _ = catalog
            .load_table(&ident)
            .await
            .expect("reload after invalidate");
        assert_eq!(
            cache.stats().body_fetches,
            1,
            "load after invalidate must body-GET (fail closed)"
        );
        assert_eq!(cache.stats().misses, 1);
    }

    /// Commit-retry note (structural pin): a retryable conflict reloads via `load_table`. With the
    /// cache, a reload of an *unchanged* pointer (loser still on base) is a hit — zero extra body
    /// GET. When the winner advanced the pointer, location string inequality forces a miss (correct
    /// fail-closed). This pin covers the unchanged-pointer leg only.
    #[tokio::test]
    async fn test_fk4_1_reload_same_pointer_is_cache_hit_commit_retry_leg() {
        let cache = Arc::new(TableMetadataCache::new());
        let catalog = new_memory_catalog_with_cache(cache.clone()).await;
        let table = create_table_with_namespace(&catalog).await;
        let ident = table.identifier().clone();

        cache.reset_stats();
        // Simulate commit-retry refresh: load, load again, same pointer.
        let a = catalog.load_table(&ident).await.expect("retry load 1");
        let b = catalog.load_table(&ident).await.expect("retry load 2");
        assert_eq!(a.metadata_location(), b.metadata_location());
        assert_eq!(cache.stats().hits, 2);
        assert_eq!(
            cache.stats().body_fetches,
            0,
            "commit-retry refresh of unchanged pointer must not re-GET body"
        );
    }

    /// `drop_table` must evict the pointer-cache entry so a recycled location cannot soft-reuse.
    #[tokio::test]
    async fn test_fk4_1_drop_table_evicts_cache_entry() {
        let cache = Arc::new(TableMetadataCache::new());
        let catalog = new_memory_catalog_with_cache(cache.clone()).await;
        let table = create_table_with_namespace(&catalog).await;
        let ident = table.identifier().clone();
        let pointer = table.metadata_location().unwrap().to_string();
        assert!(cache.lookup(&pointer, None).is_some());

        catalog.drop_table(&ident).await.expect("drop");
        assert!(
            cache.lookup(&pointer, None).is_none(),
            "drop_table must invalidate the metadata-location cache entry"
        );
    }

    /// Invalidating an unknown table must not thrash other tables' pointer entries
    /// (Java `invalidateTable` is a no-op for absent idents — not a session-wide clear).
    #[tokio::test]
    async fn test_fk4_1_invalidate_missing_table_does_not_clear_session() {
        let cache = Arc::new(TableMetadataCache::new());
        let catalog = new_memory_catalog_with_cache(cache.clone()).await;
        let table = create_table_with_namespace(&catalog).await;
        let pointer = table.metadata_location().unwrap().to_string();
        assert!(
            cache.lookup(&pointer, None).is_some(),
            "create must seed the cache"
        );

        let missing = TableIdent::new(NamespaceIdent::new("nope".into()), "ghost".into());
        catalog
            .invalidate_table(&missing)
            .await
            .expect("missing invalidate is Ok");
        assert!(
            cache.lookup(&pointer, None).is_some(),
            "invalidate of missing table must not clear sibling pointer entries"
        );
    }

    /// Successful update must drop the prior pointer key from the session cache.
    #[tokio::test]
    async fn test_fk4_1_update_evicts_prior_pointer() {
        let cache = Arc::new(TableMetadataCache::new());
        let catalog = new_memory_catalog_with_cache(cache.clone()).await;
        let table = create_table_with_namespace(&catalog).await;
        let ident = table.identifier().clone();
        let base = table.metadata_location().unwrap().to_string();

        let commit = TableCommit::builder()
            .ident(ident.clone())
            .requirements(vec![])
            .updates(vec![TableUpdate::SetProperties {
                updates: HashMap::from([("c6".to_string(), "1".to_string())]),
            }])
            .base_metadata_location(Some(base.clone()))
            .build();
        let updated = catalog.update_table(commit).await.expect("update");
        let new_loc = updated.metadata_location().unwrap().to_string();
        assert_ne!(base, new_loc);
        assert!(
            cache.lookup(&base, None).is_none(),
            "prior pointer must be evicted after successful update"
        );
        assert!(
            cache.lookup(&new_loc, None).is_some(),
            "new pointer must be seeded"
        );
    }
}
