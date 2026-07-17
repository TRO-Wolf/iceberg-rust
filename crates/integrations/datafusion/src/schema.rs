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

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use datafusion::catalog::SchemaProvider;
use datafusion::datasource::TableProvider;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::prelude::SessionContext;
use futures::StreamExt;
use iceberg::arrow::arrow_schema_to_schema_auto_assign_ids;
use iceberg::inspect::MetadataTableType;
use iceberg::spec::FormatVersion;
use iceberg::{Catalog, Error, ErrorKind, NamespaceIdent, Result, TableCreation, TableIdent};

use crate::table::IcebergTableProvider;
use crate::to_datafusion_error;

/// Represents a [`SchemaProvider`] for the Iceberg [`Catalog`], managing
/// access to table providers within a specific namespace.
#[derive(Debug)]
pub(crate) struct IcebergSchemaProvider {
    /// Reference to the Iceberg catalog
    catalog: Arc<dyn Catalog>,
    /// The namespace this schema represents
    namespace: NamespaceIdent,
    /// Directory of the table names known in this namespace, captured from `list_tables` at
    /// construction and updated by `register_table` / `deregister_table`.
    ///
    /// The value is the LAZILY-resolved provider: `None` until the table is first referenced,
    /// `Some` once its metadata has been loaded. A key's presence means the table was *listed* —
    /// it does NOT imply the table's metadata is loadable. The metadata read (an object-storage
    /// round-trip) is deferred to [`SchemaProvider::table`] (see [`Self::resolve_table`]), matching
    /// Java/Spark lazy-by-name resolution: a table that cannot load never fails registration nor any
    /// query that does not reference it, and errors loud — by name — only at reference. Wrapped in
    /// `Arc` to share across the async boundary in `register_table`.
    tables: Arc<DashMap<String, Option<Arc<IcebergTableProvider>>>>,
}

impl IcebergSchemaProvider {
    /// Asynchronously constructs a new [`IcebergSchemaProvider`] for the given namespace.
    ///
    /// Registration LISTS the namespace's table names only — it must NOT read any table's metadata.
    /// Each listed name is recorded with an unresolved (`None`) provider; the metadata read is
    /// deferred to [`SchemaProvider::table`] (see [`Self::resolve_table`]). This is Java/Spark
    /// lazy-by-name parity: a foreign, unreadable, or IAM-blocked table never fails registration nor
    /// any query that does not reference it (it errors loud, by name, only at reference), and
    /// construction is O(#tables to *list*) rather than O(#tables to *load*).
    pub(crate) async fn try_new(
        client: Arc<dyn Catalog>,
        namespace: NamespaceIdent,
    ) -> Result<Self> {
        // The listed name set is captured once here; a table created/dropped in the catalog after
        // construction is not reflected until the provider is rebuilt (the pre-existing staleness
        // note). Resolution of each name to a provider is lazy — see `resolve_table`.
        let table_names: Vec<_> = client
            .list_tables(&namespace)
            .await?
            .iter()
            .map(|tbl| tbl.name().to_string())
            .collect();

        let tables = Arc::new(DashMap::new());
        for name in table_names {
            tables.insert(name, None);
        }

        Ok(IcebergSchemaProvider {
            catalog: client,
            namespace,
            tables,
        })
    }

    /// Lazily resolves a base table name to its catalog-backed provider, caching the result.
    ///
    /// - An unknown name (not in this namespace's listing) returns `Ok(None)` — never a metadata read.
    /// - A known-but-unresolved name performs the ONE deferred `load_table` (the object-storage read)
    ///   via [`IcebergTableProvider::try_new`], caches the provider, and returns it.
    /// - A load failure is surfaced as a loud [`DataFusionError`] NAMING the table (propagated, never
    ///   swallowed to `Ok(None)`), and is NOT cached — a later reference re-attempts the load.
    ///
    /// The DashMap guard is dropped before the `.await` (the state is copied out first), so a shard
    /// lock is never held across the load.
    async fn resolve_table(&self, name: &str) -> DFResult<Option<Arc<IcebergTableProvider>>> {
        enum State {
            Loaded(Arc<IcebergTableProvider>),
            KnownUnresolved,
            Unknown,
        }

        let state = match self.tables.get(name) {
            Some(entry) => match entry.value() {
                Some(provider) => State::Loaded(provider.clone()),
                None => State::KnownUnresolved,
            },
            None => State::Unknown,
        };

        match state {
            State::Loaded(provider) => Ok(Some(provider)),
            State::Unknown => Ok(None),
            State::KnownUnresolved => {
                // The single deferred metadata read. A failure here is the loud, by-name error the
                // lazy contract promises — propagate it; do NOT swallow to `Ok(None)`.
                let provider = Arc::new(
                    IcebergTableProvider::try_new(
                        self.catalog.clone(),
                        self.namespace.clone(),
                        name,
                    )
                    .await
                    .map_err(to_datafusion_error)?,
                );
                // Cache the resolved provider, but only while the name is still listed — a concurrent
                // `deregister_table` must not be resurrected.
                if let Some(mut entry) = self.tables.get_mut(name) {
                    *entry.value_mut() = Some(provider.clone());
                }
                Ok(Some(provider))
            }
        }
    }
}

#[async_trait]
impl SchemaProvider for IcebergSchemaProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_names(&self) -> Vec<String> {
        self.tables
            .iter()
            .flat_map(|entry| {
                let table_name = entry.key().clone();
                [table_name.clone()]
                    .into_iter()
                    .chain(
                        MetadataTableType::all_types().map(move |metadata_table_name| {
                            format!("{}${}", table_name, metadata_table_name.as_str())
                        }),
                    )
            })
            .collect()
    }

    fn table_exist(&self, name: &str) -> bool {
        if let Some((table_name, metadata_table_name)) = name.split_once('$') {
            self.tables.contains_key(table_name)
                && MetadataTableType::try_from(metadata_table_name).is_ok()
        } else {
            self.tables.contains_key(name)
        }
    }

    async fn table(&self, name: &str) -> DFResult<Option<Arc<dyn TableProvider>>> {
        if let Some((table_name, metadata_table_name)) = name.split_once('$') {
            let metadata_table_type =
                MetadataTableType::try_from(metadata_table_name).map_err(DataFusionError::Plan)?;
            // Lazily resolve the BASE table, then build the requested metadata table over it. An
            // unloadable base errors loud (by name) here, exactly as a direct reference would.
            return match self.resolve_table(table_name).await? {
                Some(table) => {
                    let metadata_table = table
                        .metadata_table(metadata_table_type)
                        .await
                        .map_err(to_datafusion_error)?;
                    Ok(Some(Arc::new(metadata_table)))
                }
                None => Ok(None),
            };
        }

        Ok(self
            .resolve_table(name)
            .await?
            .map(|provider| provider as Arc<dyn TableProvider>))
    }

    fn register_table(
        &self,
        name: String,
        table: Arc<dyn TableProvider>,
    ) -> DFResult<Option<Arc<dyn TableProvider>>> {
        // Check if table already exists
        if self.table_exist(name.as_str()) {
            return Err(DataFusionError::Execution(format!(
                "Table {name} already exists"
            )));
        }

        // Convert DataFusion schema to Iceberg schema
        // DataFusion schemas don't have field IDs, so we use the function that assigns them automatically
        let df_schema = table.schema();
        let iceberg_schema = arrow_schema_to_schema_auto_assign_ids(df_schema.as_ref())
            .map_err(to_datafusion_error)?;

        // Create the table in the Iceberg catalog. The schema may contain v3-only types (e.g. a
        // DataFusion `TIMESTAMP` maps to Iceberg `timestamp_ns`), which a v2 table cannot hold, so
        // pick a format version that accommodates the schema — v2 by default, v3 only when required.
        let schema_min_format_version = iceberg_schema.min_format_version();
        let format_version = if schema_min_format_version > FormatVersion::V2 {
            schema_min_format_version
        } else {
            FormatVersion::V2
        };
        let table_creation = TableCreation::builder()
            .name(name.clone())
            .schema(iceberg_schema)
            .format_version(format_version)
            .build();

        let catalog = self.catalog.clone();
        let namespace = self.namespace.clone();
        let tables = self.tables.clone();
        let name_clone = name.clone();

        // Run the async catalog work on a runtime the *caller* does not own (see
        // `block_on_off_caller_runtime`). The `SchemaProvider` trait method is synchronous but must
        // do async catalog I/O; the previous `spawn_blocking` bridge panics ("no reactor running")
        // when called outside a runtime, and nesting a `block_on` on the caller's `current_thread`
        // runtime risks deadlock. Executing on a separate OS thread with its own runtime is robust
        // across every caller context.
        block_on_off_caller_runtime(async move {
            // Verify the input table is empty - CREATE TABLE only accepts schema definition
            ensure_table_is_empty(&table)
                .await
                .map_err(to_datafusion_error)?;

            catalog
                .create_table(&namespace, table_creation)
                .await
                .map_err(to_datafusion_error)?;

            // Create a new table provider using the catalog reference
            let table_provider = IcebergTableProvider::try_new(
                catalog.clone(),
                namespace.clone(),
                name_clone.clone(),
            )
            .await
            .map_err(to_datafusion_error)?;

            // Store the new table provider (already resolved).
            tables.insert(name_clone, Some(Arc::new(table_provider)));

            Ok(None)
        })
        .map_err(|e| DataFusionError::Execution(format!("Failed to create Iceberg table: {e}")))?
    }

    fn deregister_table(&self, name: &str) -> DFResult<Option<Arc<dyn TableProvider>>> {
        // Check if table exists
        if !self.table_exist(name) {
            return Ok(None);
        }

        let catalog = self.catalog.clone();
        let namespace = self.namespace.clone();
        let tables = self.tables.clone();
        let table_name = name.to_string();

        // Run on a runtime the caller does not own — see `register_table` and
        // `block_on_off_caller_runtime` for why running on the caller's runtime is unsafe (the old
        // bridge panics with no ambient runtime; nesting a `block_on` risks deadlock).
        block_on_off_caller_runtime(async move {
            let table_ident = TableIdent::new(namespace, table_name.clone());

            // Drop the table from the Iceberg catalog
            catalog
                .drop_table(&table_ident)
                .await
                .map_err(to_datafusion_error)?;

            // Remove from local cache and return the removed provider (if it had been resolved; a
            // listed-but-never-loaded entry holds `None` and yields `None` here).
            let removed = tables
                .remove(&table_name)
                .and_then(|(_, provider)| provider)
                .map(|provider| provider as Arc<dyn TableProvider>);

            Ok(removed)
        })
        .map_err(|e| DataFusionError::Execution(format!("Failed to drop Iceberg table: {e}")))?
    }
}

/// Drive `future` to completion on a Tokio runtime that the *calling* thread does not own, and
/// block the caller until it resolves.
///
/// # Why a separate OS thread
///
/// The two `SchemaProvider` mutators above are synchronous trait methods that must perform async
/// catalog I/O. They may be called from many caller contexts — from inside DataFusion's (typically
/// `current_thread`) runtime, from a `#[tokio::test]` (also `current_thread`), or from no runtime at
/// all. The previous bridge (`spawn_blocking` + `Handle::current().block_on` + an outer
/// `futures::executor::block_on`) is unsafe across that range:
///
/// * **No-ambient-runtime panic (the observed old-code failure)** — `spawn_blocking` /
///   `Handle::current()` panic with "there is no reactor running, must be called from the context
///   of a Tokio 1.x runtime" when the mutator is invoked outside any runtime. This is the concrete
///   bug this helper fixes (witnessed by `tests::..._work_without_caller_runtime`).
/// * **Nested-runtime deadlock/panic (the hazard this design also forecloses)** — driving an inner
///   future on the *caller's* `current_thread` runtime by parking its single worker can deadlock,
///   and `Handle::current().block_on(..)` / `Runtime::block_on` while already inside a runtime
///   panics ("Cannot start a runtime from within a runtime"). `tokio::task::block_in_place` is no
///   escape — it panics on a `current_thread` runtime.
///
/// Spawning a fresh OS thread (which has *no* ambient runtime) and building a small
/// `current_thread` runtime there lets us `block_on` safely from ANY caller context: the work runs
/// entirely off the caller's runtime, so the caller's thread merely waits on a `JoinHandle` and no
/// runtime is nested. The cost is one short-lived thread + runtime per DDL call, which is acceptable
/// for `register_table`/`deregister_table` (rare, schema-level operations, not a hot path).
///
/// Errors building the runtime or joining the thread are surfaced as a typed
/// [`DataFusionError`]; there are no `unwrap`/`expect` on the join or the channel.
fn block_on_off_caller_runtime<F>(future: F) -> DFResult<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    std::thread::scope(|scope| {
        let handle = scope.spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to build runtime for Iceberg catalog operation: {e}"
                    ))
                })?;
            Ok(runtime.block_on(future))
        });

        handle.join().map_err(|_| {
            DataFusionError::Execution("Iceberg catalog operation thread panicked".to_string())
        })?
    })
}

/// Verifies that a table provider contains no data by scanning with LIMIT 1.
/// Returns an error if the table has any rows.
async fn ensure_table_is_empty(table: &Arc<dyn TableProvider>) -> Result<()> {
    let session_ctx = SessionContext::new();
    let exec_plan = table
        .scan(&session_ctx.state(), None, &[], Some(1))
        .await
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Failed to scan table: {e}")))?;

    let task_ctx = Arc::new(TaskContext::default());
    let stream = exec_plan.execute(0, task_ctx).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to execute scan: {e}"),
        )
    })?;

    let batches: Vec<_> = stream.collect().await;
    let has_data = batches
        .into_iter()
        .filter_map(|r| r.ok())
        .any(|batch| batch.num_rows() > 0);

    if has_data {
        return Err(Error::new(
            ErrorKind::Unexpected,
            "register_table does not support tables with data.",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use datafusion::arrow::array::{Int32Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::datasource::MemTable;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::{Catalog, CatalogBuilder, NamespaceIdent};
    use tempfile::TempDir;

    use super::*;

    async fn create_test_schema_provider() -> (IcebergSchemaProvider, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let warehouse_path = temp_dir.path().to_str().unwrap().to_string();

        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
            )
            .await
            .unwrap();

        let namespace = NamespaceIdent::new("test_ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();

        let provider = IcebergSchemaProvider::try_new(Arc::new(catalog), namespace)
            .await
            .unwrap();

        (provider, temp_dir)
    }

    #[tokio::test]
    async fn test_register_table_with_data_fails() {
        let (schema_provider, _temp_dir) = create_test_schema_provider().await;

        // Create a MemTable with data
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
        ])
        .unwrap();

        let mem_table = MemTable::try_new(arrow_schema, vec![vec![batch]]).unwrap();

        // Attempt to register the table with data - should fail
        let result = schema_provider.register_table("test_table".to_string(), Arc::new(mem_table));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("register_table does not support tables with data."),
            "Expected error about tables with data, got: {err}",
        );
    }

    #[tokio::test]
    async fn test_register_empty_table_succeeds() {
        let (schema_provider, _temp_dir) = create_test_schema_provider().await;

        // Create an empty MemTable (schema only, no data rows)
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        // Create an empty batch (0 rows) - MemTable requires at least one partition
        let empty_batch = RecordBatch::new_empty(arrow_schema.clone());
        let mem_table = MemTable::try_new(arrow_schema, vec![vec![empty_batch]]).unwrap();

        // Attempt to register the empty table - should succeed
        let result = schema_provider.register_table("empty_table".to_string(), Arc::new(mem_table));

        assert!(result.is_ok(), "Expected success, got: {result:?}");

        // Verify the table was registered
        assert!(schema_provider.table_exist("empty_table"));
    }

    #[tokio::test]
    async fn test_register_duplicate_table_fails() {
        let (schema_provider, _temp_dir) = create_test_schema_provider().await;

        // Create empty MemTables
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let empty_batch1 = RecordBatch::new_empty(arrow_schema.clone());
        let empty_batch2 = RecordBatch::new_empty(arrow_schema.clone());
        let mem_table1 = MemTable::try_new(arrow_schema.clone(), vec![vec![empty_batch1]]).unwrap();
        let mem_table2 = MemTable::try_new(arrow_schema, vec![vec![empty_batch2]]).unwrap();

        // Register first table - should succeed
        let result1 = schema_provider.register_table("dup_table".to_string(), Arc::new(mem_table1));
        assert!(result1.is_ok());

        // Register second table with same name - should fail
        let result2 = schema_provider.register_table("dup_table".to_string(), Arc::new(mem_table2));
        assert!(result2.is_err());
        let err = result2.unwrap_err();
        assert!(
            err.to_string().contains("already exists"),
            "Expected error about table already existing, got: {err}",
        );
    }

    #[tokio::test]
    async fn test_deregister_table_succeeds() {
        let (schema_provider, _temp_dir) = create_test_schema_provider().await;

        // Create and register an empty table
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let empty_batch = RecordBatch::new_empty(arrow_schema.clone());
        let mem_table = MemTable::try_new(arrow_schema, vec![vec![empty_batch]]).unwrap();

        // Register the table
        let result = schema_provider.register_table("drop_me".to_string(), Arc::new(mem_table));
        assert!(result.is_ok());
        assert!(schema_provider.table_exist("drop_me"));

        // Deregister the table
        let result = schema_provider.deregister_table("drop_me");
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());

        // Verify the table no longer exists
        assert!(!schema_provider.table_exist("drop_me"));
    }

    #[tokio::test]
    async fn test_deregister_nonexistent_table_returns_none() {
        let (schema_provider, _temp_dir) = create_test_schema_provider().await;

        // Attempt to deregister a table that doesn't exist
        let result = schema_provider.deregister_table("nonexistent");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    /// FIX 1 — the synchronous `SchemaProvider` mutators must complete when invoked from inside a
    /// `current_thread` runtime, which is DataFusion's default and the `#[tokio::test]` default (the
    /// common production path). The off-caller-runtime helper runs the async catalog work on a
    /// separate OS thread, so the caller's single runtime worker is never the one that must drive
    /// the inner future. The `tokio::time::timeout` is a safety net: should a future regression
    /// reintroduce a nested-`block_on` that parks the only driving thread, the bound trips and the
    /// test fails loudly instead of hanging CI. (The genuine old-code failure mode this PR fixes is
    /// the no-ambient-runtime panic, witnessed by `..._work_without_caller_runtime` below; the old
    /// `spawn_blocking` bridge happened to complete on a `current_thread` runtime, so this test is a
    /// forward-looking positive/guard test, not a witness that the old code hung.)
    #[tokio::test]
    async fn test_register_and_deregister_succeed_on_current_thread_runtime() {
        let (schema_provider, _temp_dir) = create_test_schema_provider().await;

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let empty_batch = RecordBatch::new_empty(arrow_schema.clone());
        let mem_table = MemTable::try_new(arrow_schema, vec![vec![empty_batch]]).unwrap();

        // register_table from within a current-thread runtime must complete (no hang, no panic).
        let registered = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            // The body is synchronous, but running it under `timeout` lets a regression that parks
            // the driving thread blow the bound rather than hang the test process.
            async { schema_provider.register_table("registered".to_string(), Arc::new(mem_table)) },
        )
        .await
        .expect("register_table did not complete on a current_thread runtime (FIX 1 regression)");
        assert!(
            registered.is_ok(),
            "register_table returned an error: {registered:?}"
        );
        assert!(schema_provider.table_exist("registered"));

        // deregister_table likewise must complete on the same current-thread runtime.
        let deregistered = tokio::time::timeout(std::time::Duration::from_secs(30), async {
            schema_provider.deregister_table("registered")
        })
        .await
        .expect("deregister_table did not complete on a current_thread runtime (FIX 1 regression)");
        assert!(
            deregistered.expect("deregister_table errored").is_some(),
            "expected the dropped provider to be returned"
        );
        assert!(!schema_provider.table_exist("registered"));
    }

    /// Companion to the current-thread witness: the same path must also work when there is NO
    /// ambient caller runtime at all (a plain synchronous caller), proving the off-caller-runtime
    /// strategy does not depend on being invoked from inside a runtime. (A multi-thread
    /// `#[tokio::test]` would be the ideal third case, but the `iceberg-datafusion` crate does not
    /// enable tokio's `rt-multi-thread` feature for tests, and enabling it is a `Cargo.toml`
    /// change out of scope here; the helper itself builds its OWN runtime, so caller-runtime
    /// flavor is irrelevant to correctness — this no-runtime case exercises that directly.)
    #[test]
    fn test_register_and_deregister_work_without_caller_runtime() {
        // Build the provider on a throwaway current-thread runtime, then drop that runtime so the
        // synchronous trait calls below run with NO ambient runtime on the calling thread.
        let setup_rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build setup runtime");
        let (schema_provider, _temp_dir) = setup_rt.block_on(create_test_schema_provider());
        drop(setup_rt);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let empty_batch = RecordBatch::new_empty(arrow_schema.clone());
        let mem_table = MemTable::try_new(arrow_schema, vec![vec![empty_batch]]).unwrap();

        let result = schema_provider.register_table("no_rt_table".to_string(), Arc::new(mem_table));
        assert!(
            result.is_ok(),
            "register without an ambient runtime: {result:?}"
        );
        assert!(schema_provider.table_exist("no_rt_table"));

        let dropped = schema_provider.deregister_table("no_rt_table");
        assert!(
            dropped.expect("deregister errored").is_some(),
            "expected dropped provider"
        );
        assert!(!schema_provider.table_exist("no_rt_table"));
    }
}
