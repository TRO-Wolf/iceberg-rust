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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::catalog::{CatalogProvider, SchemaProvider};
use futures::{StreamExt, TryStreamExt, stream};
use iceberg::{Catalog, Error, ErrorKind, NamespaceIdent, Result};

use crate::schema::IcebergSchemaProvider;

/// Joins the levels of a multi-level [`NamespaceIdent`] into one DataFusion schema name. DataFusion
/// maps exactly one `&str` to one [`SchemaProvider`], so the level list must flatten.
/// [`NamespaceIdent::to_url_string`] already uses `U+001F`, so the fork keeps one flattening.
const NAMESPACE_SEPARATOR: char = '\u{1f}';

/// The separator of the ergonomic, non-canonical schema-name alias: a plain dot. A query cannot
/// type `U+001F`, so each multi-level namespace also gets a dot-joined alias. Java
/// `Namespace.toString` renders a namespace the same way.
const NAMESPACE_ALIAS_SEPARATOR: &str = ".";

/// Maximum number of levels a discovered namespace may have. The catalog server controls the tree,
/// so it is untrusted input. Without this cap a server that answers each listing with a deeper
/// namespace lists forever.
const MAX_NAMESPACE_DEPTH: usize = 64;

/// Maximum number of catalog listing round-trips in flight at once during discovery.
///
/// Discovery is N+1 over an unknown tree, so the fan-out needs a bound. 16 matches
/// `DEFAULT_LIST_STAT_CONCURRENCY` in `iceberg-storage-opendal`.
const NAMESPACE_DISCOVERY_CONCURRENCY: usize = 16;

/// Serves every namespace of an Iceberg [`Catalog`] as a DataFusion [`SchemaProvider`]. Each
/// namespace registers under its canonical [`NAMESPACE_SEPARATOR`]-joined name, which is what
/// [`Self::schema_names`] reports. A multi-level namespace also resolves through its dot-joined
/// alias.
#[derive(Debug)]
pub struct IcebergCatalogProvider {
    /// Canonical schema name to provider. This is what [`CatalogProvider::schema_names`] reports.
    schemas: HashMap<String, Arc<dyn SchemaProvider>>,
    /// Unambiguous dot-joined alias to provider. Resolution only, never listed.
    aliases: HashMap<String, Arc<dyn SchemaProvider>>,
}

impl IcebergCatalogProvider {
    /// Builds a schema provider for every namespace of the catalog, at every nesting level.
    /// Discovery is a breadth-first walk from `list_namespaces(None)`. To snapshot only some
    /// namespaces, use [`Self::try_new_with_namespace_scope`].
    pub async fn try_new(client: Arc<dyn Catalog>) -> Result<Self> {
        Self::try_new_with_scope(client, NamespaceWalkScope::Unscoped).await
    }

    /// Like [`Self::try_new`], but snapshots only `namespaces` and their descendants. The named
    /// identifiers seed the same walk. This call never issues `list_namespaces(None)`, so a sibling
    /// of a named root is neither listed nor registered.
    pub async fn try_new_with_namespace_scope(
        client: Arc<dyn Catalog>,
        namespaces: impl IntoIterator<Item = NamespaceIdent>,
    ) -> Result<Self> {
        Self::try_new_with_scope(
            client,
            NamespaceWalkScope::Scoped(namespaces.into_iter().collect()),
        )
        .await
    }

    async fn try_new_with_scope(
        client: Arc<dyn Catalog>,
        scope: NamespaceWalkScope,
    ) -> Result<Self> {
        // TODO:
        // Schemas and providers should be cached and evicted based on time
        // As of right now; schemas might become stale.
        let namespaces = discover_namespaces(&client, scope).await?;
        let providers = build_schema_providers(&client, &namespaces).await?;

        // `discover_namespaces` de-duplicates and rejects a level holding the separator, so
        // this join is injective and no insert overwrites another namespace's provider.
        let mut schemas: HashMap<String, Arc<dyn SchemaProvider>> =
            HashMap::with_capacity(namespaces.len());
        for (namespace, provider) in namespaces.iter().zip(providers.iter()) {
            schemas.insert(canonical_schema_name(namespace), provider.clone());
        }

        // Count first, so an alias two namespaces claim is dropped for both, not won by the
        // last insert.
        let mut alias_claims: HashMap<String, usize> = HashMap::new();
        for namespace in namespaces.iter().filter(|ns| ns.len() > 1) {
            *alias_claims
                .entry(alias_schema_name(namespace))
                .or_insert(0) += 1;
        }

        let mut aliases: HashMap<String, Arc<dyn SchemaProvider>> = HashMap::new();
        for (namespace, provider) in namespaces.iter().zip(providers.iter()) {
            // A single-level namespace's alias is its canonical name.
            if namespace.len() <= 1 {
                continue;
            }
            let alias = alias_schema_name(namespace);
            // Canonical bindings win outright, and an alias two namespaces both claim is dropped.
            if schemas.contains_key(&alias) || alias_claims.get(&alias).copied().unwrap_or(0) > 1 {
                continue;
            }
            aliases.insert(alias, provider.clone());
        }

        Ok(IcebergCatalogProvider { schemas, aliases })
    }
}

impl CatalogProvider for IcebergCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        // Canonical names only. Listing the aliases too doubles every nested namespace in
        // `information_schema`.
        self.schemas.keys().cloned().collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas
            .get(name)
            .or_else(|| self.aliases.get(name))
            .cloned()
    }
}

/// The identity-preserving schema name for `namespace`. [`NAMESPACE_SEPARATOR`] has the inverse.
fn canonical_schema_name(namespace: &NamespaceIdent) -> String {
    namespace.to_url_string()
}

/// The SQL-typeable alias for `namespace`. Not injective, see [`NAMESPACE_ALIAS_SEPARATOR`].
fn alias_schema_name(namespace: &NamespaceIdent) -> String {
    namespace.as_ref().join(NAMESPACE_ALIAS_SEPARATOR)
}

/// Rejects a namespace that cannot render as an identity-preserving schema name.
///
/// Both arms fail loud rather than skip. A dropped namespace is silently absent. An accepted
/// separator inside a level lets two namespaces render to one canonical name.
fn validate_namespace_renderable(namespace: &NamespaceIdent) -> Result<()> {
    if namespace.len() > MAX_NAMESPACE_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Iceberg namespace {:?} is nested {} levels deep, beyond the supported maximum of {MAX_NAMESPACE_DEPTH}",
                namespace.as_ref(),
                namespace.len(),
            ),
        ));
    }

    if let Some(level) = namespace
        .as_ref()
        .iter()
        .find(|level| level.contains(NAMESPACE_SEPARATOR))
    {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Iceberg namespace {:?} cannot be exposed to DataFusion: level {level:?} contains \
                 the U+001F namespace separator, so its schema name would be ambiguous with a \
                 differently nested namespace",
                namespace.as_ref(),
            ),
        ));
    }

    Ok(())
}

/// Where the namespace filter enters without changing [`CatalogProvider`].
///
/// `Unscoped` is the full walk. `Scoped` seeds the same walk at the named identifiers and never
/// lists the catalog root. An empty scoped list walks nothing.
enum NamespaceWalkScope {
    Unscoped,
    Scoped(Vec<NamespaceIdent>),
}

/// Breadth-first walk of the namespace tree, optionally seeded at a named scope. Two guarantees
/// terminate the walk, and neither replaces the other. `seen` expands a namespace at most once, the
/// only defence against a cycle and against a server that ignores the parent filter.
async fn discover_namespaces(
    client: &Arc<dyn Catalog>,
    scope: NamespaceWalkScope,
) -> Result<Vec<NamespaceIdent>> {
    let mut seen: HashSet<NamespaceIdent> = HashSet::new();
    let mut discovered: Vec<NamespaceIdent> = Vec::new();
    let (mut frontier, scope_seeds): (Vec<NamespaceIdent>, Option<Vec<NamespaceIdent>>) =
        match scope {
            NamespaceWalkScope::Unscoped => (client.list_namespaces(None).await?, None),
            NamespaceWalkScope::Scoped(seeds) if seeds.is_empty() => return Ok(Vec::new()),
            NamespaceWalkScope::Scoped(seeds) => (seeds.clone(), Some(seeds)),
        };

    while !frontier.is_empty() {
        let mut fresh: Vec<NamespaceIdent> = Vec::with_capacity(frontier.len());
        for namespace in frontier {
            if let Some(seeds) = scope_seeds.as_deref()
                && !seeds
                    .iter()
                    .any(|seed| namespace_is_or_under(seed, &namespace))
            {
                continue;
            }
            validate_namespace_renderable(&namespace)?;
            if seen.insert(namespace.clone()) {
                discovered.push(namespace.clone());
                fresh.push(namespace);
            }
        }

        if fresh.is_empty() {
            break;
        }

        frontier = list_child_namespaces(client, &fresh).await?;
    }

    Ok(discovered)
}

/// True when `candidate` is `seed` or a descendant of `seed` (seed levels are a prefix).
fn namespace_is_or_under(seed: &NamespaceIdent, candidate: &NamespaceIdent) -> bool {
    let seed = seed.as_ref();
    let candidate = candidate.as_ref();
    candidate.len() >= seed.len()
        && candidate
            .iter()
            .zip(seed.iter())
            .all(|(left, right)| left == right)
}

/// Lists the direct children of every namespace in `parents`.
///
/// A failing listing aborts the whole walk and names the parent. See
/// [`IcebergCatalogProvider::try_new`] for why the failure is loud.
async fn list_child_namespaces(
    client: &Arc<dyn Catalog>,
    parents: &[NamespaceIdent],
) -> Result<Vec<NamespaceIdent>> {
    let nested: Vec<Vec<NamespaceIdent>> = stream::iter(parents.iter().map(|parent| async move {
        client.list_namespaces(Some(parent)).await.map_err(|err| {
            Error::new(
                ErrorKind::Unexpected,
                format!(
                    "Failed to list the child namespaces of Iceberg namespace {:?}",
                    parent.as_ref()
                ),
            )
            .with_source(err)
        })
    }))
    .buffer_unordered(NAMESPACE_DISCOVERY_CONCURRENCY)
    .try_collect()
    .await?;

    Ok(nested.into_iter().flatten().collect())
}

/// Builds one [`IcebergSchemaProvider`] per discovered namespace, in the same order.
///
/// The caller zips the result against `namespaces`, so order is load-bearing. This uses ordered
/// `buffered`. An unordered buffer binds providers to the wrong schema names.
async fn build_schema_providers(
    client: &Arc<dyn Catalog>,
    namespaces: &[NamespaceIdent],
) -> Result<Vec<Arc<dyn SchemaProvider>>> {
    stream::iter(namespaces.iter().map(|namespace| {
        let client = client.clone();
        let namespace = namespace.clone();
        async move {
            IcebergSchemaProvider::try_new(client, namespace.clone())
                .await
                .map(|provider| Arc::new(provider) as Arc<dyn SchemaProvider>)
                .map_err(|err| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "Failed to build a DataFusion schema provider for Iceberg namespace {:?}",
                            namespace.as_ref()
                        ),
                    )
                    .with_source(err)
                })
        }
    }))
    .buffered(NAMESPACE_DISCOVERY_CONCURRENCY)
    .try_collect()
    .await
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    use iceberg::table::Table;
    use iceberg::{
        Catalog, CatalogBuilder, MemoryCatalog, Namespace, NamespaceIdent, Result, TableCommit,
        TableCreation, TableIdent,
    };
    use tempfile::TempDir;

    use super::*;

    /// `["a", "b"]` rendered canonically.
    fn canonical(levels: &[&str]) -> String {
        levels.join("\u{1f}")
    }

    async fn memory_catalog() -> (Arc<MemoryCatalog>, TempDir) {
        let temp_dir = TempDir::new().expect("failed to create the warehouse temp dir");
        let warehouse = temp_dir
            .path()
            .to_str()
            .expect("warehouse path is not valid UTF-8")
            .to_string();

        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse)]),
            )
            .await
            .expect("failed to build the in-memory catalog");

        (Arc::new(catalog), temp_dir)
    }

    /// Creates `levels` as a namespace, assuming every ancestor already exists.
    async fn create_namespace(catalog: &Arc<MemoryCatalog>, levels: &[&str]) -> NamespaceIdent {
        let namespace =
            NamespaceIdent::from_strs(levels).expect("failed to build the namespace identifier");
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap_or_else(|e| panic!("failed to create namespace {levels:?}: {e}"));
        namespace
    }

    /// Creates `levels` and every ancestor of it.
    async fn create_namespace_chain(catalog: &Arc<MemoryCatalog>, levels: &[&str]) {
        for depth in 1..=levels.len() {
            let prefix = &levels[..depth];
            let namespace =
                NamespaceIdent::from_strs(prefix).expect("failed to build the namespace ident");
            if !catalog
                .namespace_exists(&namespace)
                .await
                .expect("namespace_exists failed")
            {
                create_namespace(catalog, prefix).await;
            }
        }
    }

    async fn create_table(catalog: &Arc<MemoryCatalog>, namespace: &NamespaceIdent, name: &str) {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .expect("failed to build the table schema");

        catalog
            .create_table(
                namespace,
                TableCreation::builder()
                    .name(name.to_string())
                    .schema(schema)
                    .build(),
            )
            .await
            .unwrap_or_else(|e| panic!("failed to create table {name}: {e}"));
    }

    /// The table names a schema provider reports, without the `table$metadata` variants.
    fn base_table_names(provider: &Arc<dyn SchemaProvider>) -> Vec<String> {
        let mut names: Vec<String> = provider
            .table_names()
            .into_iter()
            .filter(|name| !name.contains('$'))
            .collect();
        names.sort();
        names
    }

    fn sorted_schema_names(provider: &IcebergCatalogProvider) -> Vec<String> {
        let mut names = provider.schema_names();
        names.sort();
        names
    }

    /// The listing budget the non-delegating [`ParentScript`] arms enforce.
    ///
    /// A walk that never terminates hangs the test binary, and a hung test reports nothing. This
    /// budget turns the hang into a typed `Err` the assertions can observe. A correct walk here
    /// spends 3 listings, so 64 cannot produce a false RED.
    const MAX_SCRIPTED_LISTINGS: usize = 64;

    /// How [`ScriptedCatalog`] answers `list_namespaces`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ParentScript {
        /// Pass the parent through to the in-memory catalog. A well-behaved server.
        Delegate,
        /// Answer every child listing with the root list. A REST server does this when it drops
        /// an unrecognised `?parent=`.
        IgnoreParent,
        /// Answer with the next root in a ring. A genuine cycle, whose depth never increases.
        Cycle,
    }

    /// Delegates everything to an in-memory catalog but scripts one listing to fail, to delay,
    /// or to report a tree that does not terminate.
    ///
    /// The failing arms pin the loud failure policy. The delay arm completes provider
    /// construction out of request order, which makes the ordered `buffered` observable. The
    /// [`ParentScript`] arms pin `seen` as what terminates a walk whose depth never grows.
    #[derive(Debug)]
    struct ScriptedCatalog {
        inner: Arc<MemoryCatalog>,
        fail_children_of: Option<NamespaceIdent>,
        fail_tables_of: Option<NamespaceIdent>,
        /// When set with [`Self::failing_tables_of`], fail every `list_tables` of that
        /// namespace. When set with [`Self::failing_tables_once_of`], fail only the first call.
        fail_tables_only_once: bool,
        fail_tables_fired: AtomicUsize,
        delay_tables_of: Option<NamespaceIdent>,
        parent_script: ParentScript,
        /// Every `list_namespaces` call. Pins the N+1 shape and feeds the listing budget.
        listing_calls: AtomicUsize,
        /// Parent of each `list_namespaces` call, in order. `None` is the catalog-root listing
        /// that a scoped walk must never issue.
        listing_parents: Mutex<Vec<Option<NamespaceIdent>>>,
        /// Namespace of each `list_tables` call, in issuance order.
        table_listings: Mutex<Vec<NamespaceIdent>>,
    }

    impl ScriptedCatalog {
        fn delegating(inner: Arc<MemoryCatalog>) -> Self {
            Self {
                inner,
                fail_children_of: None,
                fail_tables_of: None,
                fail_tables_only_once: false,
                fail_tables_fired: AtomicUsize::new(0),
                delay_tables_of: None,
                parent_script: ParentScript::Delegate,
                listing_calls: AtomicUsize::new(0),
                listing_parents: Mutex::new(Vec::new()),
                table_listings: Mutex::new(Vec::new()),
            }
        }

        fn failing_children_of(inner: Arc<MemoryCatalog>, namespace: NamespaceIdent) -> Self {
            Self {
                fail_children_of: Some(namespace),
                ..Self::delegating(inner)
            }
        }

        fn failing_tables_of(inner: Arc<MemoryCatalog>, namespace: NamespaceIdent) -> Self {
            Self {
                fail_tables_of: Some(namespace),
                ..Self::delegating(inner)
            }
        }

        fn failing_tables_once_of(inner: Arc<MemoryCatalog>, namespace: NamespaceIdent) -> Self {
            Self {
                fail_tables_of: Some(namespace),
                fail_tables_only_once: true,
                ..Self::delegating(inner)
            }
        }

        fn delaying_tables_of(inner: Arc<MemoryCatalog>, namespace: NamespaceIdent) -> Self {
            Self {
                delay_tables_of: Some(namespace),
                ..Self::delegating(inner)
            }
        }

        fn scripting_parents(inner: Arc<MemoryCatalog>, parent_script: ParentScript) -> Self {
            Self {
                parent_script,
                ..Self::delegating(inner)
            }
        }

        fn listings_issued(&self) -> usize {
            self.listing_calls.load(Ordering::Relaxed)
        }

        fn recorded_listing_parents(&self) -> Vec<Option<NamespaceIdent>> {
            self.listing_parents
                .lock()
                .expect("listing_parents lock poisoned")
                .clone()
        }

        fn recorded_table_listings(&self) -> Vec<NamespaceIdent> {
            self.table_listings
                .lock()
                .expect("table_listings lock poisoned")
                .clone()
        }
    }

    #[async_trait]
    impl Catalog for ScriptedCatalog {
        async fn list_namespaces(
            &self,
            parent: Option<&NamespaceIdent>,
        ) -> Result<Vec<NamespaceIdent>> {
            let calls = self.listing_calls.fetch_add(1, Ordering::Relaxed) + 1;
            self.listing_parents
                .lock()
                .expect("listing_parents lock poisoned")
                .push(parent.cloned());
            if self.parent_script != ParentScript::Delegate && calls > MAX_SCRIPTED_LISTINGS {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "namespace discovery issued more than {MAX_SCRIPTED_LISTINGS} listings \
                         against a {:?} catalog: the walk is not terminating",
                        self.parent_script
                    ),
                ));
            }

            if let (Some(parent), Some(target)) = (parent, self.fail_children_of.as_ref())
                && parent == target
            {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    "simulated child-listing failure",
                ));
            }

            let mut namespaces = match self.parent_script {
                ParentScript::Delegate => self.inner.list_namespaces(parent).await?,
                ParentScript::IgnoreParent => self.inner.list_namespaces(None).await?,
                ParentScript::Cycle => {
                    let mut roots = self.inner.list_namespaces(None).await?;
                    roots.sort();
                    match parent {
                        // Enter the ring at its first element only.
                        None => roots.into_iter().take(1).collect(),
                        Some(parent) => match roots.iter().position(|root| root == parent) {
                            Some(index) => vec![roots[(index + 1) % roots.len()].clone()],
                            None => Vec::new(),
                        },
                    }
                }
            };
            // Sorted so the request order of provider construction is deterministic. The memory
            // catalog otherwise yields hash order, and the out-of-order test then only sometimes
            // tells `buffered` from `buffer_unordered`.
            namespaces.sort();
            Ok(namespaces)
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
            self.table_listings
                .lock()
                .expect("table_listings lock poisoned")
                .push(namespace.clone());
            if self.fail_tables_of.as_ref() == Some(namespace) {
                let fired = self.fail_tables_fired.fetch_add(1, Ordering::Relaxed);
                if !self.fail_tables_only_once || fired == 0 {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        "simulated table-listing failure",
                    ));
                }
            }
            if self.delay_tables_of.as_ref() == Some(namespace) {
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
            self.inner.list_tables(namespace).await
        }

        async fn create_table(
            &self,
            namespace: &NamespaceIdent,
            creation: TableCreation,
        ) -> Result<Table> {
            self.inner.create_table(namespace, creation).await
        }

        async fn load_table(&self, table: &TableIdent) -> Result<Table> {
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

        async fn register_table(
            &self,
            table: &TableIdent,
            metadata_location: String,
        ) -> Result<Table> {
            self.inner.register_table(table, metadata_location).await
        }

        async fn update_table(&self, commit: TableCommit) -> Result<Table> {
            self.inner.update_table(commit).await
        }
    }

    /// The single-level shape: one schema per namespace, named as before nested discovery.
    #[tokio::test]
    async fn single_level_namespaces_are_unchanged() {
        let (catalog, _dir) = memory_catalog().await;
        let ns1 = create_namespace(&catalog, &["ns1"]).await;
        let ns2 = create_namespace(&catalog, &["ns2"]).await;
        create_table(&catalog, &ns1, "t1").await;
        create_table(&catalog, &ns2, "t2").await;

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("catalog provider construction failed");

        assert_eq!(sorted_schema_names(&provider), vec![
            "ns1".to_string(),
            "ns2".to_string()
        ]);
        let ns1_provider = provider.schema("ns1").expect("ns1 schema missing");
        assert_eq!(base_table_names(&ns1_provider), vec!["t1".to_string()]);
        let ns2_provider = provider.schema("ns2").expect("ns2 schema missing");
        assert_eq!(base_table_names(&ns2_provider), vec!["t2".to_string()]);
    }

    /// The multi-level namespace `["a", "b"]` survives as one schema with its own tables, not
    /// as two bare schemas `a` and `b`.
    ///
    /// Mutation: restore the `.flat_map(|ns| ns.as_ref().clone())` explosion. The bare `"b"`
    /// assertion and the canonical lookup both go RED.
    #[tokio::test]
    async fn multi_level_namespace_keeps_its_identity_and_tables() {
        let (catalog, _dir) = memory_catalog().await;
        let parent = create_namespace(&catalog, &["a"]).await;
        let child = create_namespace(&catalog, &["a", "b"]).await;
        create_table(&catalog, &parent, "t_in_a").await;
        create_table(&catalog, &child, "t_in_a_b").await;

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("catalog provider construction failed");

        assert_eq!(sorted_schema_names(&provider), vec![
            "a".to_string(),
            canonical(&["a", "b"])
        ]);

        let child_provider = provider
            .schema(&canonical(&["a", "b"]))
            .expect("the nested namespace has no schema provider");
        assert_eq!(base_table_names(&child_provider), vec![
            "t_in_a_b".to_string()
        ]);

        let parent_provider = provider.schema("a").expect("the parent schema is missing");
        assert_eq!(base_table_names(&parent_provider), vec![
            "t_in_a".to_string()
        ]);

        // The exploded component must not exist as a schema of its own.
        assert!(
            provider.schema("b").is_none(),
            "the bare level \"b\" was registered as a schema; the namespace was flattened"
        );
    }

    /// Discovery recurses, and it recurses past two levels.
    ///
    /// Mutation: delete the `frontier = list_child_namespaces(..)` line, so only the root list
    /// is walked. The three-level canonical lookup goes RED.
    #[tokio::test]
    async fn namespaces_nested_three_levels_deep_are_discovered() {
        let (catalog, _dir) = memory_catalog().await;
        create_namespace_chain(&catalog, &["a", "b", "c"]).await;
        let deepest = NamespaceIdent::from_strs(["a", "b", "c"]).expect("bad namespace");
        create_table(&catalog, &deepest, "deep_table").await;

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("catalog provider construction failed");

        assert_eq!(sorted_schema_names(&provider), vec![
            "a".to_string(),
            canonical(&["a", "b"]),
            canonical(&["a", "b", "c"]),
        ]);

        let deep_provider = provider
            .schema(&canonical(&["a", "b", "c"]))
            .expect("the three-level namespace has no schema provider");
        assert_eq!(base_table_names(&deep_provider), vec![
            "deep_table".to_string()
        ]);
    }

    /// The canonical name round-trips back to the exact namespace.
    ///
    /// Mutation: join `canonical_schema_name` with `.`. The recovered identifier for
    /// `["a.b", "c"]` becomes `["a", "b", "c"]` and the assertion goes RED.
    #[tokio::test]
    async fn canonical_schema_names_invert_to_the_exact_namespace() {
        let (catalog, _dir) = memory_catalog().await;
        create_namespace(&catalog, &["a.b"]).await;
        create_namespace(&catalog, &["a.b", "c"]).await;

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("catalog provider construction failed");

        for name in provider.schema_names() {
            let recovered = NamespaceIdent::from_strs(name.split(NAMESPACE_SEPARATOR))
                .expect("a canonical schema name failed to parse back");
            assert_eq!(
                canonical_schema_name(&recovered),
                name,
                "the canonical name {name:?} did not round-trip"
            );
            assert!(
                provider.schema(&name).is_some(),
                "the recovered namespace {recovered:?} has no schema"
            );
        }
        assert_eq!(sorted_schema_names(&provider), vec![
            "a.b".to_string(),
            canonical(&["a.b", "c"]),
        ]);
    }

    /// The alias resolves a nested namespace to the same provider as its canonical name.
    ///
    /// Mutation: delete the `aliases` lookup arm from `CatalogProvider::schema`. `schema("x.y")`
    /// becomes `None` and the test goes RED.
    #[tokio::test]
    async fn dot_alias_resolves_a_nested_namespace() {
        let (catalog, _dir) = memory_catalog().await;
        create_namespace_chain(&catalog, &["x", "y"]).await;
        let nested = NamespaceIdent::from_strs(["x", "y"]).expect("bad namespace");
        create_table(&catalog, &nested, "aliased").await;

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("catalog provider construction failed");

        let via_alias = provider
            .schema("x.y")
            .expect("the dot alias did not resolve");
        assert_eq!(base_table_names(&via_alias), vec!["aliased".to_string()]);

        // The alias resolves only. It must not appear in the listed schema names.
        assert!(
            !provider.schema_names().contains(&"x.y".to_string()),
            "the alias leaked into schema_names()"
        );
    }

    /// Collision 1: the alias of `["a", "b"]` is also the canonical name of `["a.b"]`. The
    /// canonical binding wins, and both namespaces stay reachable under their canonical names.
    ///
    /// Mutation: drop the `schemas.contains_key(&alias)` guard so the alias inserts anyway. The
    /// two table-name assertions below then go RED.
    #[tokio::test]
    async fn alias_never_shadows_a_canonical_schema_name() {
        let (catalog, _dir) = memory_catalog().await;
        let dotted = create_namespace(&catalog, &["a.b"]).await;
        create_namespace_chain(&catalog, &["a", "b"]).await;
        let nested = NamespaceIdent::from_strs(["a", "b"]).expect("bad namespace");
        create_table(&catalog, &dotted, "in_dotted").await;
        create_table(&catalog, &nested, "in_nested").await;

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("catalog provider construction failed");

        // Sorted byte-wise: U+001F (31) sorts before '.' (46).
        assert_eq!(sorted_schema_names(&provider), vec![
            "a".to_string(),
            canonical(&["a", "b"]),
            "a.b".to_string(),
        ]);

        // `a.b` is the single-level namespace's own name, so it serves that namespace's table.
        let dotted_provider = provider
            .schema("a.b")
            .expect("the dotted schema is missing");
        assert_eq!(base_table_names(&dotted_provider), vec![
            "in_dotted".to_string()
        ]);

        // The nested namespace keeps its identity under the canonical name.
        let nested_provider = provider
            .schema(&canonical(&["a", "b"]))
            .expect("the nested schema is missing");
        assert_eq!(base_table_names(&nested_provider), vec![
            "in_nested".to_string()
        ]);

        assert!(
            !provider.aliases.contains_key("a.b"),
            "an alias was registered over a canonical schema name"
        );
    }

    /// Collision 2: two different nested namespaces claim the alias `a.b.c`. Neither wins, so a
    /// query fails loud instead of reading the wrong namespace.
    ///
    /// Mutation: remove the `alias_claims.get(&alias) > 1` guard. The alias resolves to the last
    /// insert and `schema("a.b.c").is_none()` goes RED.
    #[tokio::test]
    async fn an_alias_claimed_by_two_namespaces_is_dropped() {
        let (catalog, _dir) = memory_catalog().await;
        create_namespace_chain(&catalog, &["a", "b", "c"]).await;
        create_namespace(&catalog, &["a", "b.c"]).await;

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("catalog provider construction failed");

        // Sorted byte-wise: U+001F (31) sorts before '.' (46).
        assert_eq!(sorted_schema_names(&provider), vec![
            "a".to_string(),
            canonical(&["a", "b"]),
            canonical(&["a", "b", "c"]),
            canonical(&["a", "b.c"]),
        ]);

        assert!(
            provider.schema("a.b.c").is_none(),
            "the ambiguous alias a.b.c resolved instead of being dropped"
        );

        // Both namespaces stay reachable under their canonical names.
        assert!(provider.schema(&canonical(&["a", "b", "c"])).is_some());
        assert!(provider.schema(&canonical(&["a", "b.c"])).is_some());
    }

    /// Collision 3: a level that contains the canonical separator makes the join non-injective.
    /// Construction fails with a typed error and names the namespace.
    ///
    /// Mutation: delete the `level.contains(NAMESPACE_SEPARATOR)` arm of
    /// `validate_namespace_renderable`. One namespace overwrites the other and `is_err()` is RED.
    #[tokio::test]
    async fn a_separator_inside_a_namespace_level_is_a_typed_error() {
        let (catalog, _dir) = memory_catalog().await;
        create_namespace(&catalog, &["a\u{1f}b"]).await;
        create_namespace_chain(&catalog, &["a", "b"]).await;

        let err = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect_err("a separator-bearing namespace level must fail construction");

        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "wrong error kind: {err}"
        );
        assert!(
            err.to_string().contains("U+001F namespace separator"),
            "the error does not explain the collision: {err}"
        );
    }

    /// The depth cap is a loud failure, not a truncation.
    ///
    /// Mutation: relax the guard to `namespace.len() > MAX_NAMESPACE_DEPTH + 1`. The over-deep
    /// namespace is accepted and `expect_err` goes RED.
    #[tokio::test]
    async fn a_namespace_deeper_than_the_cap_fails_loudly() {
        let (catalog, _dir) = memory_catalog().await;
        let levels: Vec<String> = (0..=MAX_NAMESPACE_DEPTH).map(|i| format!("l{i}")).collect();
        let level_refs: Vec<&str> = levels.iter().map(String::as_str).collect();
        create_namespace_chain(&catalog, &level_refs).await;

        let err = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect_err("a namespace nested beyond the cap must fail construction");

        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "wrong error kind: {err}"
        );
        assert!(
            err.to_string().contains("beyond the supported maximum"),
            "the error does not name the depth cap: {err}"
        );
    }

    /// Builds a two-root catalog whose child listings follow `parent_script`, then discovers
    /// under a wall-clock timeout so a non-terminating walk fails instead of wedging the binary.
    /// The returned warehouse dir must outlive the provider and the catalog.
    async fn discover_under_script(
        parent_script: ParentScript,
    ) -> (IcebergCatalogProvider, Arc<ScriptedCatalog>, TempDir) {
        let (inner, warehouse) = memory_catalog().await;
        let ns1 = create_namespace(&inner, &["ns1"]).await;
        let ns2 = create_namespace(&inner, &["ns2"]).await;
        create_table(&inner, &ns1, "t1").await;
        create_table(&inner, &ns2, "t2").await;

        let scripted = Arc::new(ScriptedCatalog::scripting_parents(inner, parent_script));
        let catalog: Arc<dyn Catalog> = scripted.clone();

        let provider = tokio::time::timeout(
            Duration::from_secs(30),
            IcebergCatalogProvider::try_new(catalog),
        )
        .await
        .unwrap_or_else(|_| {
            panic!("namespace discovery against a {parent_script:?} catalog never terminated")
        })
        .unwrap_or_else(|e| {
            panic!("catalog provider construction failed against a {parent_script:?} catalog: {e}")
        });

        (provider, scripted, warehouse)
    }

    /// Termination 1: a server answers every child listing with the root list, which a REST
    /// catalog does when it drops an unrecognised `?parent=`. Depth never grows, so `seen` is
    /// the only thing that stops the walk.
    ///
    /// Mutation: delete the `seen.insert` guard in [`discover_namespaces`]. The walk re-lists
    /// the same two namespaces forever, the listing budget trips, and construction gives RED.
    #[tokio::test]
    async fn a_catalog_that_ignores_the_parent_filter_still_terminates() {
        let (provider, scripted, _warehouse) =
            discover_under_script(ParentScript::IgnoreParent).await;

        assert_eq!(sorted_schema_names(&provider), vec![
            "ns1".to_string(),
            "ns2".to_string()
        ]);
        // One root listing plus one child listing per namespace. The N+1 shape, walked once.
        assert_eq!(
            scripted.listings_issued(),
            3,
            "the walk re-expanded an already-visited namespace"
        );
    }

    /// Termination 2: a genuine cycle, where the catalog reports `ns1`'s child as `ns2` and
    /// `ns2`'s child as `ns1`. Depth never grows, so only `seen` terminates the walk.
    ///
    /// Mutation: delete the `seen.insert` guard. The ring walks forever and gives RED.
    /// Second mutation: move `let mut seen = HashSet::new()` inside the `while` loop. The
    /// frontier is one namespace wide here, so termination needs `seen` to survive across rounds.
    #[tokio::test]
    async fn a_cyclic_catalog_still_terminates() {
        let (provider, scripted, _warehouse) = discover_under_script(ParentScript::Cycle).await;

        assert_eq!(sorted_schema_names(&provider), vec![
            "ns1".to_string(),
            "ns2".to_string()
        ]);
        // Root listing, then one child listing for each of the two namespaces in the ring.
        assert_eq!(
            scripted.listings_issued(),
            3,
            "the walk went round the cycle more than once"
        );
    }

    /// A namespace whose child listing fails must fail construction and name itself.
    ///
    /// Mutation: swallow the error in `list_child_namespaces` with `.unwrap_or_default()`.
    /// `expect_err` goes RED.
    #[tokio::test]
    async fn a_child_listing_failure_fails_construction_and_names_the_namespace() {
        let (inner, _dir) = memory_catalog().await;
        create_namespace_chain(&inner, &["good"]).await;
        create_namespace_chain(&inner, &["blocked", "hidden"]).await;
        let blocked = NamespaceIdent::from_strs(["blocked"]).expect("bad namespace");

        let catalog: Arc<dyn Catalog> =
            Arc::new(ScriptedCatalog::failing_children_of(inner, blocked));

        let err = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect_err("a failing child listing must fail construction");

        assert!(
            err.to_string().contains("child namespaces") && err.to_string().contains("blocked"),
            "the error does not name the namespace that could not be listed: {err}"
        );
    }

    /// A namespace whose table listing fails must not fail construction. `table_names` and
    /// `table_exist` return empty without caching the failure. `table()` fails loud.
    ///
    /// Mutation: make `list_tables` eager in [`IcebergSchemaProvider::try_new`]. Construction
    /// then errors and this test goes RED.
    #[tokio::test]
    async fn a_table_listing_failure_is_deferred_to_first_result_bearing_access() {
        let (inner, _dir) = memory_catalog().await;
        create_namespace_chain(&inner, &["good"]).await;
        create_namespace_chain(&inner, &["broken", "leaf"]).await;
        let broken = NamespaceIdent::from_strs(["broken", "leaf"]).expect("bad namespace");

        let scripted = Arc::new(ScriptedCatalog::failing_tables_of(inner, broken.clone()));
        let catalog: Arc<dyn Catalog> = scripted.clone();

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("list_tables failure must not fail catalog construction");
        assert!(
            scripted.recorded_table_listings().is_empty(),
            "construction issued list_tables: {:?}",
            scripted.recorded_table_listings()
        );

        let broken_name = canonical(&["broken", "leaf"]);
        let broken_schema = provider
            .schema(&broken_name)
            .expect("broken namespace must still be registered");
        assert!(
            base_table_names(&broken_schema).is_empty(),
            "table_names must swallow a listing failure"
        );
        assert!(
            !broken_schema.table_exist("anything"),
            "table_exist must swallow a listing failure"
        );
        let err = broken_schema
            .table("anything")
            .await
            .expect_err("table() must surface the listing failure");
        assert!(
            err.to_string().contains("simulated table-listing failure")
                || err.to_string().contains("table-listing"),
            "the error does not carry the listing failure: {err}"
        );
        assert!(
            scripted.recorded_table_listings().contains(&broken),
            "first access must have issued list_tables({broken:?})"
        );
    }

    /// Providers must stay aligned with the namespaces they zip against even when the catalog
    /// answers out of request order. Otherwise a query reads another namespace's tables.
    /// [`ScriptedCatalog`] lists namespaces sorted, so `a_delayed` is requested first and its
    /// sleeping table listing completes last.
    #[tokio::test]
    async fn providers_stay_bound_to_their_own_namespace_when_listings_finish_out_of_order() {
        let (inner, _dir) = memory_catalog().await;
        let delayed = create_namespace(&inner, &["a_delayed"]).await;
        let prompt = create_namespace(&inner, &["b_prompt"]).await;
        create_table(&inner, &delayed, "delayed_table").await;
        create_table(&inner, &prompt, "prompt_table").await;

        let catalog: Arc<dyn Catalog> =
            Arc::new(ScriptedCatalog::delaying_tables_of(inner, delayed));

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("catalog provider construction failed");

        let delayed_provider = provider
            .schema("a_delayed")
            .expect("a_delayed schema missing");
        assert_eq!(base_table_names(&delayed_provider), vec![
            "delayed_table".to_string()
        ]);
        let prompt_provider = provider
            .schema("b_prompt")
            .expect("b_prompt schema missing");
        assert_eq!(base_table_names(&prompt_provider), vec![
            "prompt_table".to_string()
        ]);
    }

    /// An unknown schema name resolves to nothing through either map.
    #[tokio::test]
    async fn an_unknown_schema_name_resolves_to_none() {
        let (catalog, _dir) = memory_catalog().await;
        create_namespace_chain(&catalog, &["a", "b"]).await;

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("catalog provider construction failed");

        assert!(provider.schema("nope").is_none());
        assert!(provider.schema("a.b.c").is_none());
        assert!(provider.schema(&canonical(&["a", "z"])).is_none());
    }

    /// Two sibling roots, `keep` and `other`. `keep` is three levels deep, so a one-level walk
    /// cannot satisfy a descendants pin. Each listed namespace has a table.
    async fn scoped_walk_fixture() -> (
        Arc<ScriptedCatalog>,
        NamespaceIdent,
        NamespaceIdent,
        NamespaceIdent,
        NamespaceIdent,
        NamespaceIdent,
        TempDir,
    ) {
        let (inner, warehouse) = memory_catalog().await;
        let keep = create_namespace(&inner, &["keep"]).await;
        let keep_child = create_namespace(&inner, &["keep", "child"]).await;
        let keep_grand = create_namespace(&inner, &["keep", "child", "grand"]).await;
        let other = create_namespace(&inner, &["other"]).await;
        let other_child = create_namespace(&inner, &["other", "child"]).await;
        create_table(&inner, &keep, "t_keep").await;
        create_table(&inner, &keep_child, "t_keep_child").await;
        create_table(&inner, &keep_grand, "t_keep_grand").await;
        create_table(&inner, &other, "t_other").await;
        create_table(&inner, &other_child, "t_other_child").await;
        (
            Arc::new(ScriptedCatalog::delegating(inner)),
            keep,
            keep_child,
            keep_grand,
            other,
            other_child,
            warehouse,
        )
    }

    /// Unscoped `try_new` lists the catalog root and every namespace, siblings included.
    ///
    /// Mutation: seed `try_new` with an empty scope. The root-listing and `other` schema
    /// assertions both go RED.
    #[tokio::test]
    async fn unscoped_try_new_still_walks_the_whole_catalog() {
        let (scripted, keep, keep_child, keep_grand, other, other_child, _warehouse) =
            scoped_walk_fixture().await;
        let catalog: Arc<dyn Catalog> = scripted.clone();

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("unscoped catalog provider construction failed");

        assert_eq!(sorted_schema_names(&provider), vec![
            "keep".to_string(),
            canonical(&["keep", "child"]),
            canonical(&["keep", "child", "grand"]),
            "other".to_string(),
            canonical(&["other", "child"]),
        ]);
        assert!(
            scripted
                .recorded_listing_parents()
                .iter()
                .any(|parent| parent.is_none()),
            "unscoped try_new must still issue list_namespaces(None): {:?}",
            scripted.recorded_listing_parents()
        );
        assert!(
            scripted.recorded_table_listings().is_empty(),
            "unscoped try_new must not list tables at construction: {:?}",
            scripted.recorded_table_listings()
        );
        for (schema_name, namespace) in [
            ("keep".to_string(), &keep),
            (canonical(&["keep", "child"]), &keep_child),
            (canonical(&["keep", "child", "grand"]), &keep_grand),
            ("other".to_string(), &other),
            (canonical(&["other", "child"]), &other_child),
        ] {
            let schema = provider
                .schema(&schema_name)
                .unwrap_or_else(|| panic!("{schema_name} schema missing"));
            let _ = base_table_names(&schema);
            assert!(
                scripted.recorded_table_listings().contains(namespace),
                "first table_names of {schema_name} never listed {namespace:?}: {:?}",
                scripted.recorded_table_listings()
            );
        }
    }

    /// A named scope walks that namespace and its descendants only. Nested names stay canonical
    /// and the dot alias still resolves.
    ///
    /// Mutation: implement scope as a post-filter on a full walk. The root-listing and `other`
    /// assertions go RED. Deleting the walk under the seed reds the child assertions.
    #[tokio::test]
    async fn scoped_walk_touches_only_the_named_namespace_and_descendants() {
        let (scripted, keep, keep_child, keep_grand, other, other_child, _warehouse) =
            scoped_walk_fixture().await;
        let catalog: Arc<dyn Catalog> = scripted.clone();

        let provider =
            IcebergCatalogProvider::try_new_with_namespace_scope(catalog, [keep.clone()])
                .await
                .expect("scoped catalog provider construction failed");

        assert_eq!(sorted_schema_names(&provider), vec![
            "keep".to_string(),
            canonical(&["keep", "child"]),
            canonical(&["keep", "child", "grand"]),
        ]);
        assert!(provider.schema("other").is_none());
        assert!(provider.schema(&canonical(&["other", "child"])).is_none());

        let keep_provider = provider.schema("keep").expect("keep schema missing");
        assert_eq!(base_table_names(&keep_provider), vec!["t_keep".to_string()]);
        let keep_child_provider = provider
            .schema(&canonical(&["keep", "child"]))
            .expect("keep.child schema missing");
        assert_eq!(base_table_names(&keep_child_provider), vec![
            "t_keep_child".to_string()
        ]);
        let keep_grand_provider = provider
            .schema(&canonical(&["keep", "child", "grand"]))
            .expect("keep.child.grand schema missing");
        assert_eq!(base_table_names(&keep_grand_provider), vec![
            "t_keep_grand".to_string()
        ]);
        let via_alias = provider
            .schema("keep.child.grand")
            .expect("the grandchild's dot alias did not resolve");
        assert_eq!(base_table_names(&via_alias), vec![
            "t_keep_grand".to_string()
        ]);

        let listing_parents = scripted.recorded_listing_parents();
        assert!(
            listing_parents.iter().all(|parent| parent.is_some()),
            "scoped walk issued list_namespaces(None): {listing_parents:?}"
        );
        assert!(
            !listing_parents
                .iter()
                .any(|parent| parent.as_ref() == Some(&other)),
            "scoped walk listed children of the sibling {other:?}: {listing_parents:?}"
        );
        assert!(
            listing_parents
                .iter()
                .any(|parent| parent.as_ref() == Some(&keep)),
            "scoped walk never listed children of the named root {keep:?}: {listing_parents:?}"
        );
        assert!(
            listing_parents
                .iter()
                .any(|parent| parent.as_ref() == Some(&keep_child)),
            "scoped walk never listed children of the descendant {keep_child:?}: {listing_parents:?}"
        );
        assert!(
            listing_parents
                .iter()
                .any(|parent| parent.as_ref() == Some(&keep_grand)),
            "scoped walk never listed children of the grandchild {keep_grand:?}: {listing_parents:?}"
        );

        let table_listings = scripted.recorded_table_listings();
        assert!(
            table_listings.contains(&keep)
                && table_listings.contains(&keep_child)
                && table_listings.contains(&keep_grand),
            "scoped walk missed a table listing inside the scope: {table_listings:?}"
        );
        assert!(
            !table_listings.contains(&other) && !table_listings.contains(&other_child),
            "scoped walk listed tables outside the scope: {table_listings:?}"
        );
    }

    /// Scoping to a nested namespace pulls in neither its parent nor a sibling tree.
    #[tokio::test]
    async fn scoped_walk_of_a_nested_namespace_excludes_its_parent() {
        let (scripted, keep, keep_child, keep_grand, other, _other_child, _warehouse) =
            scoped_walk_fixture().await;
        let catalog: Arc<dyn Catalog> = scripted.clone();

        let provider =
            IcebergCatalogProvider::try_new_with_namespace_scope(catalog, [keep_child.clone()])
                .await
                .expect("scoped catalog provider construction failed");

        assert_eq!(sorted_schema_names(&provider), vec![
            canonical(&["keep", "child"]),
            canonical(&["keep", "child", "grand"]),
        ]);
        assert!(provider.schema("keep").is_none());
        assert!(provider.schema("other").is_none());
        assert_eq!(
            base_table_names(
                &provider
                    .schema(&canonical(&["keep", "child", "grand"]))
                    .expect("grandchild of the named nested scope missing")
            ),
            vec!["t_keep_grand".to_string()]
        );

        let listing_parents = scripted.recorded_listing_parents();
        assert!(
            listing_parents.iter().all(|parent| parent.is_some()),
            "nested scoped walk issued list_namespaces(None): {listing_parents:?}"
        );
        assert!(
            !listing_parents
                .iter()
                .any(|parent| parent.as_ref() == Some(&keep) || parent.as_ref() == Some(&other)),
            "nested scoped walk listed a namespace outside the named subtree: {listing_parents:?}"
        );
        assert!(
            listing_parents
                .iter()
                .any(|parent| parent.as_ref() == Some(&keep_grand)),
            "nested scoped walk never listed children of the in-scope grandchild: {listing_parents:?}"
        );
        assert!(
            !scripted.recorded_table_listings().contains(&keep),
            "nested scoped walk listed tables of the excluded parent"
        );
    }

    /// An empty scope walks nothing. It does not fall back to the full catalog.
    ///
    /// Mutation: treat an empty iterator as `Unscoped`. The provider registers `keep` and
    /// `other`, and `schema_names` is non-empty.
    #[tokio::test]
    async fn empty_scope_walks_nothing() {
        let (scripted, _keep, _keep_child, _keep_grand, _other, _other_child, _warehouse) =
            scoped_walk_fixture().await;
        let catalog: Arc<dyn Catalog> = scripted.clone();

        let provider = IcebergCatalogProvider::try_new_with_namespace_scope(catalog, [])
            .await
            .expect("empty-scope construction failed");

        assert!(
            provider.schema_names().is_empty(),
            "empty scope registered schemas: {:?}",
            provider.schema_names()
        );
        assert_eq!(
            scripted.listings_issued(),
            0,
            "empty scope issued list_namespaces: {:?}",
            scripted.recorded_listing_parents()
        );
        assert!(
            scripted.recorded_table_listings().is_empty(),
            "empty scope issued list_tables: {:?}",
            scripted.recorded_table_listings()
        );
    }

    /// Two named roots are both walked, still without a catalog-root listing.
    #[tokio::test]
    async fn scoped_walk_accepts_multiple_named_roots() {
        let (scripted, keep, _keep_child, _keep_grand, other, _other_child, _warehouse) =
            scoped_walk_fixture().await;
        let catalog: Arc<dyn Catalog> = scripted.clone();

        let provider = IcebergCatalogProvider::try_new_with_namespace_scope(catalog, [
            keep.clone(),
            other.clone(),
        ])
        .await
        .expect("multi-root scoped construction failed");

        assert_eq!(sorted_schema_names(&provider), vec![
            "keep".to_string(),
            canonical(&["keep", "child"]),
            canonical(&["keep", "child", "grand"]),
            "other".to_string(),
            canonical(&["other", "child"]),
        ]);
        assert!(
            scripted
                .recorded_listing_parents()
                .iter()
                .all(|parent| parent.is_some()),
            "multi-root scoped walk issued list_namespaces(None)"
        );
    }

    /// A server that ignores `?parent=` must still not register a sibling of the named scope.
    /// Without the descendant filter the scoped walk becomes a full catalog snapshot.
    ///
    /// Mutation: delete the `namespace_is_or_under` guard. `ns2` appears in `schema_names` and
    /// `list_tables(ns2)` is issued.
    #[tokio::test]
    async fn scoped_walk_rejects_siblings_when_the_catalog_ignores_the_parent_filter() {
        let (inner, _warehouse) = memory_catalog().await;
        let ns1 = create_namespace(&inner, &["ns1"]).await;
        let ns2 = create_namespace(&inner, &["ns2"]).await;
        create_table(&inner, &ns1, "t1").await;
        create_table(&inner, &ns2, "t2").await;

        let scripted = Arc::new(ScriptedCatalog::scripting_parents(
            inner,
            ParentScript::IgnoreParent,
        ));
        let catalog: Arc<dyn Catalog> = scripted.clone();

        let provider = tokio::time::timeout(
            Duration::from_secs(30),
            IcebergCatalogProvider::try_new_with_namespace_scope(catalog, [ns1.clone()]),
        )
        .await
        .unwrap_or_else(|_| {
            panic!("scoped discovery against an IgnoreParent catalog never terminated")
        })
        .expect("scoped construction failed against an IgnoreParent catalog");

        assert_eq!(sorted_schema_names(&provider), vec!["ns1".to_string()]);
        assert!(provider.schema("ns2").is_none());
        assert!(
            !scripted.recorded_table_listings().contains(&ns2),
            "scoped walk listed tables of a sibling the parent-ignoring catalog re-reported"
        );
        assert!(
            scripted
                .recorded_listing_parents()
                .iter()
                .all(|parent| parent.is_some()),
            "scoped IgnoreParent walk issued list_namespaces(None)"
        );
    }

    /// A sibling outside the scope is never listed, so its failing children never surface.
    ///
    /// Mutation: implement scope as a full walk then a filter. Construction then fails on
    /// `blocked` the way unscoped `try_new` does.
    #[tokio::test]
    async fn scoped_walk_does_not_observe_a_sibling_listing_failure() {
        let (inner, _dir) = memory_catalog().await;
        let good = create_namespace(&inner, &["good"]).await;
        create_namespace_chain(&inner, &["blocked", "hidden"]).await;
        let blocked = NamespaceIdent::from_strs(["blocked"]).expect("bad namespace");
        create_table(&inner, &good, "t_good").await;

        let catalog: Arc<dyn Catalog> =
            Arc::new(ScriptedCatalog::failing_children_of(inner, blocked));

        let provider = IcebergCatalogProvider::try_new_with_namespace_scope(catalog, [good])
            .await
            .expect(
                "a sibling listing failure must not be observed inside another namespace's scope",
            );

        assert_eq!(sorted_schema_names(&provider), vec!["good".to_string()]);
    }

    /// Construction issues zero `list_tables`. The first `table_names` issues one, a later
    /// `table_names` issues none.
    #[tokio::test]
    async fn list_tables_is_lazy_per_namespace_and_cached_on_success() {
        let (scripted, keep, _keep_child, _keep_grand, _other, _other_child, _warehouse) =
            scoped_walk_fixture().await;
        let catalog: Arc<dyn Catalog> = scripted.clone();

        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("catalog provider construction failed");
        assert!(
            scripted.recorded_table_listings().is_empty(),
            "construction listed tables: {:?}",
            scripted.recorded_table_listings()
        );

        let keep_schema = provider.schema("keep").expect("keep schema missing");
        assert_eq!(base_table_names(&keep_schema), vec!["t_keep".to_string()]);
        let after_first = scripted
            .recorded_table_listings()
            .iter()
            .filter(|namespace| *namespace == &keep)
            .count();
        assert_eq!(
            after_first, 1,
            "first table_names must list keep exactly once"
        );

        assert_eq!(base_table_names(&keep_schema), vec!["t_keep".to_string()]);
        let after_second = scripted
            .recorded_table_listings()
            .iter()
            .filter(|namespace| *namespace == &keep)
            .count();
        assert_eq!(
            after_second,
            1,
            "successful listing must be cached: {:?}",
            scripted.recorded_table_listings()
        );
    }

    /// A failed `list_tables` is not cached. The next access retries and can succeed.
    #[tokio::test]
    async fn a_failed_list_tables_is_retried_on_the_next_access() {
        let (inner, _dir) = memory_catalog().await;
        let ns = create_namespace(&inner, &["retry"]).await;
        create_table(&inner, &ns, "t_retry").await;

        let scripted = Arc::new(ScriptedCatalog::failing_tables_once_of(inner, ns.clone()));
        let catalog: Arc<dyn Catalog> = scripted.clone();
        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect("construction must ignore the pending table-listing failure");

        let schema = provider.schema("retry").expect("retry schema missing");
        assert!(
            base_table_names(&schema).is_empty(),
            "first table_names must swallow the one-shot failure"
        );
        assert_eq!(
            scripted
                .recorded_table_listings()
                .iter()
                .filter(|namespace| *namespace == &ns)
                .count(),
            1
        );

        assert_eq!(base_table_names(&schema), vec!["t_retry".to_string()]);
        assert_eq!(
            scripted
                .recorded_table_listings()
                .iter()
                .filter(|namespace| *namespace == &ns)
                .count(),
            2,
            "the second access must retry the failed listing"
        );
    }
}
