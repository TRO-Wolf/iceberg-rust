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

//! Lazy / failure-tolerant table resolution for the Iceberg DataFusion catalog provider (F-A2-2).
//!
//! Invariant under test (Java/Spark lazy-by-name parity): registering an Iceberg catalog provider
//! must NOT read any table's metadata; a table that cannot load must not fail registration nor any
//! query that does not reference it; the failing table itself must error loud, by name, at
//! reference time.
//!
//! Every pin drives the PUBLIC `IcebergCatalogProvider` surface with a mock [`Catalog`] whose
//! `load_table` is faulted, entirely offline: an in-memory catalog backend over local-FS storage,
//! no AWS, no network.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use datafusion::catalog::CatalogProvider;
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::table::Table;
use iceberg::{
    Catalog, CatalogBuilder, Error, ErrorKind, MemoryCatalog, Namespace, NamespaceIdent, Result,
    TableCommit, TableCreation, TableIdent,
};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

// =================================================================================================
// Offline fixtures
// =================================================================================================

/// A fresh temp warehouse directory; the returned [`TempDir`] guard must be kept for the test's
/// lifetime (dropping it deletes the directory).
fn temp_warehouse() -> (String, TempDir) {
    let dir = TempDir::new().expect("create temp dir");
    let path = dir
        .path()
        .to_str()
        .expect("temp path is valid utf-8")
        .to_string();
    (path, dir)
}

/// An in-memory catalog backed by local-FS storage rooted at `warehouse` (offline).
async fn memory_catalog(warehouse: &str) -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("build memory catalog")
}

/// A `{foo1 int, foo2 string}` table creation at `location`.
fn table_creation(location: &str, name: &str) -> TableCreation {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build schema");
    TableCreation::builder()
        .location(location.to_string())
        .name(name.to_string())
        .properties(HashMap::new())
        .schema(schema)
        .build()
}

/// Sum of the row counts across a query's result batches.
async fn row_count(ctx: &SessionContext, sql: &str) -> usize {
    let batches = ctx
        .sql(sql)
        .await
        .expect("plan query")
        .collect()
        .await
        .expect("execute query");
    batches.iter().map(|b| b.num_rows()).sum()
}

// =================================================================================================
// Fault-injecting `Catalog` wrapper
// =================================================================================================

/// Which `load_table` calls the wrapper faults.
#[derive(Debug)]
enum LoadFault {
    /// EVERY `load_table` errors — proves registration performs ZERO eager metadata loads.
    All,
    /// Only these table names error on load; all others delegate — proves a good table coexists
    /// with an unloadable one.
    Only(HashSet<String>),
}

/// A [`Catalog`] that delegates every operation to `inner` EXCEPT `load_table`, which is faulted
/// per [`LoadFault`]. `load_calls` counts every `load_table` invocation so a test can assert that
/// registration triggered none. The injected error NAMES the table, so a reference-time failure is
/// loud and attributable — exactly what a foreign / IAM-blocked / corrupt table produces in
/// production.
#[derive(Debug)]
struct FaultyLoadCatalog {
    inner: Arc<dyn Catalog>,
    fault: LoadFault,
    load_calls: AtomicUsize,
}

impl FaultyLoadCatalog {
    fn new(inner: Arc<dyn Catalog>, fault: LoadFault) -> Self {
        Self {
            inner,
            fault,
            load_calls: AtomicUsize::new(0),
        }
    }

    fn load_calls(&self) -> usize {
        self.load_calls.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl Catalog for FaultyLoadCatalog {
    async fn list_namespaces(
        &self,
        parent: Option<&NamespaceIdent>,
    ) -> Result<Vec<NamespaceIdent>> {
        self.inner.list_namespaces(parent).await
    }

    async fn create_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<Namespace> {
        self.inner.create_namespace(namespace, properties).await
    }

    async fn get_namespace(&self, namespace: &NamespaceIdent) -> Result<Namespace> {
        self.inner.get_namespace(namespace).await
    }

    async fn namespace_exists(&self, namespace: &NamespaceIdent) -> Result<bool> {
        self.inner.namespace_exists(namespace).await
    }

    async fn update_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<()> {
        self.inner.update_namespace(namespace, properties).await
    }

    async fn drop_namespace(&self, namespace: &NamespaceIdent) -> Result<()> {
        self.inner.drop_namespace(namespace).await
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        self.inner.list_tables(namespace).await
    }

    async fn create_table(
        &self,
        namespace: &NamespaceIdent,
        creation: TableCreation,
    ) -> Result<Table> {
        self.inner.create_table(namespace, creation).await
    }

    /// The faulted method: counts the call, then errors (naming the table) per [`LoadFault`].
    async fn load_table(&self, table: &TableIdent) -> Result<Table> {
        self.load_calls.fetch_add(1, Ordering::SeqCst);
        let faulted = match &self.fault {
            LoadFault::All => true,
            LoadFault::Only(names) => names.contains(table.name()),
        };
        if faulted {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("injected load_table failure for table '{}'", table.name()),
            ));
        }
        self.inner.load_table(table).await
    }

    async fn drop_table(&self, table: &TableIdent) -> Result<()> {
        self.inner.drop_table(table).await
    }

    async fn table_exists(&self, table: &TableIdent) -> Result<bool> {
        self.inner.table_exists(table).await
    }

    async fn rename_table(&self, src: &TableIdent, dest: &TableIdent) -> Result<()> {
        self.inner.rename_table(src, dest).await
    }

    async fn register_table(&self, table: &TableIdent, metadata_location: String) -> Result<Table> {
        self.inner.register_table(table, metadata_location).await
    }

    async fn update_table(&self, commit: TableCommit) -> Result<Table> {
        self.inner.update_table(commit).await
    }
}

// =================================================================================================
// Pins (one per F-A2-2 enumeration element)
// =================================================================================================

/// P1 (load-bearing) — registering a catalog whose `load_table` ALWAYS errors must SUCCEED, and
/// must perform ZERO table-metadata loads. One unloadable table can no longer brick session
/// construction, and startup cost is O(#tables to list), not O(#tables to load).
///
/// MUTATION (restore eager per-table construction in `IcebergSchemaProvider::try_new`): `try_new`
/// would `load_table` every listed table, the always-fault would propagate, registration would
/// return `Err`, and the `expect` below fails RED.
#[tokio::test]
async fn registration_tolerates_always_failing_load_table() -> Result<()> {
    let (wh, _guard) = temp_warehouse();
    let catalog = memory_catalog(&wh).await;
    let ns = NamespaceIdent::new("ns".to_string());
    catalog.create_namespace(&ns, HashMap::new()).await?;
    // Three real (schema-only) tables so `list_tables` returns their names.
    for name in ["t_a", "t_b", "t_c"] {
        catalog
            .create_table(&ns, table_creation(&format!("{wh}/{name}"), name))
            .await?;
    }
    let inner: Arc<dyn Catalog> = Arc::new(catalog);

    let mock = Arc::new(FaultyLoadCatalog::new(inner, LoadFault::All));
    let provider = IcebergCatalogProvider::try_new(mock.clone())
        .await
        .expect("registration must succeed even when every load_table errors (lazy resolution)");

    assert_eq!(
        mock.load_calls(),
        0,
        "registration must read ZERO table metadata (no eager load_table calls)"
    );

    // The namespace and all three names are still visible — from listing, not loading.
    let schema = provider
        .schema("ns")
        .expect("namespace schema present after registration");
    assert!(schema.table_exist("t_a"));
    assert!(schema.table_exist("t_b"));
    assert!(schema.table_exist("t_c"));
    assert_eq!(
        mock.load_calls(),
        0,
        "listing/existence checks must also perform no load_table calls"
    );
    Ok(())
}

/// P2 — a good table queries fine while an unloadable table coexists in the SAME namespace.
///
/// MUTATION (restore eager construction): registration fails on the unloadable table before any
/// query runs — the mock `try_new` `expect` fails RED.
#[tokio::test]
async fn good_table_queries_while_unloadable_table_coexists() -> Result<()> {
    let (wh, _guard) = temp_warehouse();
    let catalog = memory_catalog(&wh).await;
    let ns = NamespaceIdent::new("ns".to_string());
    catalog.create_namespace(&ns, HashMap::new()).await?;
    catalog
        .create_table(&ns, table_creation(&format!("{wh}/good"), "good"))
        .await?;
    catalog
        .create_table(
            &ns,
            table_creation(&format!("{wh}/unloadable"), "unloadable"),
        )
        .await?;
    let inner: Arc<dyn Catalog> = Arc::new(catalog);

    // Seed the good table with two rows through a WORKING provider over the inner catalog.
    let seed_ctx = SessionContext::new();
    seed_ctx.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(inner.clone()).await?),
    );
    seed_ctx
        .sql("INSERT INTO catalog.ns.good VALUES (1, 'a'), (2, 'b')")
        .await
        .expect("plan insert")
        .collect()
        .await
        .expect("execute insert into good");

    // Register over a catalog whose load_table FAULTS only for `unloadable`.
    let mock = Arc::new(FaultyLoadCatalog::new(
        inner,
        LoadFault::Only(HashSet::from(["unloadable".to_string()])),
    ));
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(
            IcebergCatalogProvider::try_new(mock)
                .await
                .expect("registration succeeds despite the unloadable table"),
        ),
    );

    // The good table scans and returns exactly its two seeded rows while `unloadable` coexists.
    let rows = row_count(&ctx, "SELECT foo1 FROM catalog.ns.good ORDER BY foo1").await;
    assert_eq!(
        rows, 2,
        "the good table returns exactly its two seeded rows"
    );
    Ok(())
}

/// P3 — referencing the unloadable table errors LOUD, NAMING the table; the session stays usable
/// (a subsequent query of the good table still works).
///
/// MUTATION (make the lazy path swallow the load error → `Ok(None)`): the reference would surface
/// as "table not found" rather than the named load error — the `expect_err` + name assertion RED.
#[tokio::test]
async fn referencing_unloadable_table_errors_by_name_and_session_stays_usable() -> Result<()> {
    let (wh, _guard) = temp_warehouse();
    let catalog = memory_catalog(&wh).await;
    let ns = NamespaceIdent::new("ns".to_string());
    catalog.create_namespace(&ns, HashMap::new()).await?;
    catalog
        .create_table(&ns, table_creation(&format!("{wh}/good"), "good"))
        .await?;
    catalog
        .create_table(
            &ns,
            table_creation(&format!("{wh}/unloadable"), "unloadable"),
        )
        .await?;
    let inner: Arc<dyn Catalog> = Arc::new(catalog);

    // Seed `good` so the post-error query returns a row.
    let seed_ctx = SessionContext::new();
    seed_ctx.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(inner.clone()).await?),
    );
    seed_ctx
        .sql("INSERT INTO catalog.ns.good VALUES (7, 'g')")
        .await
        .expect("plan insert")
        .collect()
        .await
        .expect("execute insert into good");

    let mock = Arc::new(FaultyLoadCatalog::new(
        inner,
        LoadFault::Only(HashSet::from(["unloadable".to_string()])),
    ));
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(mock)
            .await
            .expect("registration succeeds"),
    );

    // Direct provider-surface resolution: `table("unloadable")` must be a loud Err NAMING the table.
    let schema = provider.schema("ns").expect("namespace schema present");
    let resolved = schema.table("unloadable").await;
    let err = resolved.expect_err("referencing an unloadable table must be a loud error, not None");
    assert!(
        err.to_string().contains("unloadable"),
        "the reference error must NAME the failing table, got: {err}"
    );

    // The good table still resolves through the same schema — the bad table did not poison it.
    let good = schema
        .table("good")
        .await
        .expect("good table resolves without error")
        .expect("good table is present");
    assert_eq!(
        good.schema().fields().len(),
        2,
        "good table schema resolved (foo1, foo2)"
    );

    // And a full SQL query of the good table still runs after the failed reference (session usable).
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", provider);
    let rows = row_count(&ctx, "SELECT foo1 FROM catalog.ns.good").await;
    assert_eq!(
        rows, 1,
        "the good table returns its seeded row after the unloadable-table reference errored"
    );
    Ok(())
}

/// P4 + P5 — `table_names()` lists BOTH the good and the unloadable table (names come from
/// listing, not loading); `table_exist` and `schema_names` are unchanged by lazy resolution, and
/// none of these read-only surfaces trigger a metadata load.
///
/// MUTATION (restore eager construction): registration fails on the unloadable table, so the
/// provider never exists to list — the mock `try_new` `expect` fails RED.
#[tokio::test]
async fn table_names_and_existence_come_from_listing_not_loading() -> Result<()> {
    let (wh, _guard) = temp_warehouse();
    let catalog = memory_catalog(&wh).await;
    let ns = NamespaceIdent::new("ns".to_string());
    catalog.create_namespace(&ns, HashMap::new()).await?;
    catalog
        .create_table(&ns, table_creation(&format!("{wh}/good"), "good"))
        .await?;
    catalog
        .create_table(
            &ns,
            table_creation(&format!("{wh}/unloadable"), "unloadable"),
        )
        .await?;
    let inner: Arc<dyn Catalog> = Arc::new(catalog);

    let mock = Arc::new(FaultyLoadCatalog::new(
        inner,
        LoadFault::Only(HashSet::from(["unloadable".to_string()])),
    ));
    let provider = IcebergCatalogProvider::try_new(mock.clone())
        .await
        .expect("registration succeeds despite the unloadable table");

    // schema_names lists the namespace (behavior unchanged).
    assert!(
        provider.schema_names().iter().any(|n| n == "ns"),
        "schema listing must include the namespace"
    );

    let schema = provider.schema("ns").expect("namespace schema present");
    let names = schema.table_names();
    let lists = |needle: &str| names.iter().any(|n| n == needle);
    assert!(lists("good"), "listing includes the good table");
    assert!(
        lists("unloadable"),
        "listing includes the UNLOADABLE table — names come from list_tables, not load_table"
    );
    // The base + metadata-table variants are present for both (behavior unchanged).
    assert!(lists("good$snapshots"), "good's metadata tables are listed");
    assert!(
        lists("unloadable$snapshots"),
        "the unloadable table's metadata tables are listed too"
    );

    // table_exist: true for both listed tables (incl. a metadata variant), false for a stranger.
    assert!(schema.table_exist("good"));
    assert!(schema.table_exist("unloadable"));
    assert!(schema.table_exist("unloadable$snapshots"));
    assert!(!schema.table_exist("does_not_exist"));

    // None of the read-only listing / existence surfaces loaded any metadata.
    assert_eq!(
        mock.load_calls(),
        0,
        "table_names / table_exist / schema_names must trigger no load_table"
    );
    Ok(())
}
