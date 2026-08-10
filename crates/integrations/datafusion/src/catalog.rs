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

/// The separator that renders a multi-level Iceberg [`NamespaceIdent`] as the single
/// DataFusion schema-name string — ASCII **unit separator** `U+001F`.
///
/// # Why a rendering is needed at all
///
/// An Iceberg namespace is an ordered list of levels (`["a", "b"]`), while DataFusion's
/// [`CatalogProvider`] maps exactly ONE `&str` to one [`SchemaProvider`]. A multi-level namespace
/// therefore has to be flattened into one string.
///
/// # Why `U+001F`
///
/// This is not a new convention: it is the one this repository already commits to in
/// [`NamespaceIdent::to_url_string`], which the REST catalog uses for every
/// `/v1/namespaces/{ns}` path segment and the S3 Tables catalog uses for every `namespace(..)`
/// request. Matching it keeps one flattening in the fork rather than two.
///
/// # The inverse (a reader MUST be able to recover the exact namespace)
///
/// `schema_name.split(NAMESPACE_SEPARATOR)` reproduces the original level list EXACTLY, and
/// `NamespaceIdent::from_strs(schema_name.split('\u{1f}'))` reconstructs the identifier itself.
/// That is total, not best-effort, because [`validate_namespace_renderable`] REJECTS at
/// construction any namespace whose level text contains the separator — so the join is provably
/// injective over everything this provider will ever hold.
const NAMESPACE_SEPARATOR: char = '\u{1f}';

/// The separator of the *ergonomic, non-canonical* schema-name alias: a plain dot.
///
/// `U+001F` cannot be typed in SQL, so a nested namespace addressed only by its canonical name
/// would be reachable programmatically and unreachable from a query. Each multi-level namespace
/// therefore also gets a dot-joined alias (`["a", "b"]` → `a.b`), which is what Java renders a
/// namespace as (`org.apache.iceberg.catalog.Namespace.toString` joins its levels with a Guava
/// DOT `Joiner`; decoded from `iceberg-api` 1.10.0 bytecode).
///
/// A dot may legally occur INSIDE a level — Java's `Namespace.of` rejects only the null byte — so
/// the alias is NOT injective and never becomes an identity. See [`IcebergCatalogProvider`] for
/// the precedence rules that keep an alias from ever shadowing a different namespace.
const NAMESPACE_ALIAS_SEPARATOR: &str = ".";

/// Maximum number of levels a discovered namespace may have.
///
/// Namespace discovery walks a tree the *catalog server* controls, so it is untrusted input in
/// exactly the sense AGENTS.md "Recursion Safety" means: the walk is an explicit-queue BFS (no
/// stack recursion), but without a bound an adversarial catalog that answers every child listing
/// with a STRICTLY DEEPER namespace would keep producing never-before-seen identifiers and issue
/// round-trips forever.
///
/// That is the only failure mode this cap addresses. It does NOT terminate a cycle, nor a catalog
/// that re-answers with an already-visited namespace: neither increases depth, so the cap is never
/// reached. Those are terminated by the `seen` visited-set in [`discover_namespaces`] — see that
/// function for the two independent guarantees and their tests.
///
/// 64 is this crate's existing nesting bound (`physical_plan::project`'s
/// `MAX_WRITE_COMPATIBILITY_DEPTH`), and is far above anything real: Glue, HMS and S3 Tables
/// return no children at all, and the deepest namespaces seen on REST catalogs are 2–3 levels.
///
/// Exceeding it is a LOUD typed error naming the offending namespace, never a truncation — a
/// silently truncated tree would make namespaces disappear from the catalog, which is the very
/// defect this module was fixed for.
const MAX_NAMESPACE_DEPTH: usize = 64;

/// Maximum number of catalog listing round-trips in flight at once during discovery.
///
/// Recursive discovery is N+1 over the namespace tree, so the fan-out has to be bounded rather
/// than issued as one unbounded `try_join_all` over an unknown tree. 16 matches the fork's other
/// listing-concurrency default (`DEFAULT_LIST_STAT_CONCURRENCY` in `iceberg-storage-opendal`).
const NAMESPACE_DISCOVERY_CONCURRENCY: usize = 16;

/// Provides an interface to manage and access multiple schemas
/// within an Iceberg [`Catalog`].
///
/// Acts as a centralized catalog provider that aggregates
/// multiple [`SchemaProvider`], each associated with distinct namespaces.
///
/// # Namespace naming
///
/// Every namespace in the catalog — at every nesting level — is registered under its **canonical**
/// schema name, its levels joined with [`NAMESPACE_SEPARATOR`] (identical to
/// [`NamespaceIdent::to_url_string`]). Canonical names are what [`Self::schema_names`] reports,
/// and they round-trip back to the exact [`NamespaceIdent`] by splitting on the same separator.
///
/// Multi-level namespaces are ADDITIONALLY reachable through a dot-joined alias
/// ([`NAMESPACE_ALIAS_SEPARATOR`]) so they can be named in SQL. Aliases are resolution-only: they
/// are not reported by [`Self::schema_names`], and they never take precedence.
///
/// # Collisions
///
/// Two rules keep one namespace from ever being served in place of another:
///
/// 1. **Canonical names cannot collide.** A namespace whose level text contains
///    [`NAMESPACE_SEPARATOR`] is rejected by [`Self::try_new`] with a typed
///    [`ErrorKind::DataInvalid`] error, which makes the canonical join injective. (`["a\u{1f}b"]`
///    and `["a", "b"]` would otherwise both render `a\u{1f}b`.)
/// 2. **An ambiguous alias is dropped, never resolved.** `["a", "b"]` and `["a.b"]` both alias to
///    `a.b`. An alias is registered only when it is unclaimed: if it equals ANY namespace's
///    canonical name, the canonical binding wins and the alias is not registered; if two distinct
///    namespaces produce the same alias, neither gets it. A dropped alias resolves to `None` —
///    the query fails to find a schema instead of silently reading the wrong one.
#[derive(Debug)]
pub struct IcebergCatalogProvider {
    /// Canonical schema name ([`NAMESPACE_SEPARATOR`]-joined) → provider. Authoritative; this is
    /// what [`CatalogProvider::schema_names`] reports.
    schemas: HashMap<String, Arc<dyn SchemaProvider>>,
    /// Unambiguous dot-joined aliases → provider. Resolution-only, never listed, never shadowing
    /// a canonical name.
    aliases: HashMap<String, Arc<dyn SchemaProvider>>,
}

impl IcebergCatalogProvider {
    /// Asynchronously tries to construct a new [`IcebergCatalogProvider`]
    /// using the given client to fetch and initialize schema providers for
    /// every namespace in the Iceberg [`Catalog`], at every nesting level.
    ///
    /// Discovery is a breadth-first walk: `list_namespaces(None)` for the roots, then
    /// `list_namespaces(Some(parent))` for each namespace found, until no new namespaces appear.
    /// The walk is bounded by [`MAX_NAMESPACE_DEPTH`] and issues at most
    /// [`NAMESPACE_DISCOVERY_CONCURRENCY`] listings concurrently.
    ///
    /// # Failure policy — loud, never partial
    ///
    /// A namespace that cannot be listed, or whose [`SchemaProvider`] cannot be built, FAILS this
    /// call with a typed error that names the namespace; it is never dropped from the catalog.
    /// The alternative the charter allows — skipping it with a `tracing` warning — is not
    /// available here: `iceberg-datafusion` does not depend on `tracing` (adding a dependency is
    /// out of scope), so a skip would be *silent*, and a namespace that silently vanishes is
    /// indistinguishable from one that does not exist. Failing loudly also preserves the
    /// pre-existing behaviour of this constructor, so no caller regresses.
    ///
    /// Note this is a per-*namespace* policy and does not weaken the lazy, failure-tolerant
    /// per-*table* contract documented in `docs/ENGINE_CONTRACT.md` §1: table metadata is still
    /// never read here, so an unreadable table still cannot brick construction.
    pub async fn try_new(client: Arc<dyn Catalog>) -> Result<Self> {
        // TODO:
        // Schemas and providers should be cached and evicted based on time
        // As of right now; schemas might become stale.
        let namespaces = discover_namespaces(&client).await?;
        let providers = build_schema_providers(&client, &namespaces).await?;

        // Canonical bindings. `discover_namespaces` de-duplicates the identifiers and rejects any
        // level containing the separator, so this join is injective and no insert can overwrite
        // a different namespace's provider.
        let mut schemas: HashMap<String, Arc<dyn SchemaProvider>> =
            HashMap::with_capacity(namespaces.len());
        for (namespace, provider) in namespaces.iter().zip(providers.iter()) {
            schemas.insert(canonical_schema_name(namespace), provider.clone());
        }

        // Alias bindings. Count first, so an alias claimed by two namespaces is dropped for BOTH
        // rather than won by whichever is inserted last.
        let mut alias_claims: HashMap<String, usize> = HashMap::new();
        for namespace in namespaces.iter().filter(|ns| ns.len() > 1) {
            *alias_claims
                .entry(alias_schema_name(namespace))
                .or_insert(0) += 1;
        }

        let mut aliases: HashMap<String, Arc<dyn SchemaProvider>> = HashMap::new();
        for (namespace, provider) in namespaces.iter().zip(providers.iter()) {
            // A single-level namespace's alias IS its canonical name; nothing to add.
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
        // Canonical names only — the aliases are alternative spellings of schemas already listed
        // here, and reporting both would double every nested namespace in `information_schema`.
        self.schemas.keys().cloned().collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas
            .get(name)
            .or_else(|| self.aliases.get(name))
            .cloned()
    }
}

/// The identity-preserving schema name for `namespace`. See [`NAMESPACE_SEPARATOR`] for the
/// inverse.
fn canonical_schema_name(namespace: &NamespaceIdent) -> String {
    namespace.to_url_string()
}

/// The ergonomic, non-canonical, SQL-typeable alias for `namespace`. Not injective — see
/// [`NAMESPACE_ALIAS_SEPARATOR`].
fn alias_schema_name(namespace: &NamespaceIdent) -> String {
    namespace.as_ref().join(NAMESPACE_ALIAS_SEPARATOR)
}

/// Rejects a namespace that cannot be rendered as an identity-preserving schema name.
///
/// Both arms are hard, typed [`ErrorKind::DataInvalid`] failures rather than skips: dropping the
/// namespace would make it silently absent, and accepting a level that contains the separator
/// would let two different namespaces render to the same canonical name.
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

/// Breadth-first walk of the whole namespace tree.
///
/// Termination rests on two guarantees that are NOT interchangeable — each covers a shape the
/// other cannot:
///
/// 1. **`seen`** — a namespace is expanded at most once, so the walk finishes on any catalog that
///    answers a child listing with an already-visited namespace. That covers a genuine cycle
///    (`a → b → a`) and a server that ignores the parent filter and re-answers with the root list,
///    which is what a REST catalog does when it drops an unrecognised `?parent=` query parameter.
///    Neither shape ever increases depth, so [`MAX_NAMESPACE_DEPTH`] is never reached and `seen`
///    is the SOLE defence; without it the constructor hangs and floods the catalog with
///    round-trips. Pinned by `a_catalog_that_ignores_the_parent_filter_still_terminates` and
///    `a_cyclic_catalog_still_terminates`.
/// 2. **[`MAX_NAMESPACE_DEPTH`]** — fails loudly on a catalog that keeps answering with strictly
///    deeper, never-before-seen namespaces, which `seen` alone would follow forever.
async fn discover_namespaces(client: &Arc<dyn Catalog>) -> Result<Vec<NamespaceIdent>> {
    let mut seen: HashSet<NamespaceIdent> = HashSet::new();
    let mut discovered: Vec<NamespaceIdent> = Vec::new();
    let mut frontier: Vec<NamespaceIdent> = client.list_namespaces(None).await?;

    while !frontier.is_empty() {
        let mut fresh: Vec<NamespaceIdent> = Vec::with_capacity(frontier.len());
        for namespace in frontier {
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

/// Lists the direct children of every namespace in `parents`, at most
/// [`NAMESPACE_DISCOVERY_CONCURRENCY`] listings in flight.
///
/// A failing listing aborts the whole walk with a typed error naming the parent — see
/// [`IcebergCatalogProvider::try_new`] for why the failure is loud rather than skipped.
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

/// Builds one [`IcebergSchemaProvider`] per discovered namespace, in the same order, at most
/// [`NAMESPACE_DISCOVERY_CONCURRENCY`] in flight.
///
/// `buffered` (ordered) rather than `buffer_unordered`: the result is zipped back against
/// `namespaces`, so order is load-bearing — an unordered buffer would bind providers to the wrong
/// schema names.
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
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

    /// The table names a schema provider reports, excluding the `table$metadata` variants that
    /// [`IcebergSchemaProvider::table_names`] also emits.
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
    /// A catalog that never terminates the BFS would otherwise HANG the test binary rather than
    /// fail it, and a hung test reports nothing. Refusing further listings past this budget turns
    /// the non-termination into a typed `Err` that the assertions can observe. Every scripted tree
    /// here is 1–2 namespaces, so a correct walk spends 3 listings; 64 leaves two orders of
    /// magnitude of headroom before the budget can produce a false RED.
    const MAX_SCRIPTED_LISTINGS: usize = 64;

    /// How [`ScriptedCatalog`] answers `list_namespaces`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ParentScript {
        /// Pass the parent through to the in-memory catalog — a well-behaved server.
        Delegate,
        /// Answer EVERY child listing with the root list, ignoring the parent. This is what a REST
        /// catalog server does when it drops an unrecognised `?parent=` query parameter.
        IgnoreParent,
        /// Answer with the next root in a ring: `ns1 → ns2 → ns1 → …`. A genuine cycle, in which
        /// the reported depth never increases.
        Cycle,
    }

    /// A catalog that delegates everything to an in-memory catalog but scripts ONE listing:
    /// either it fails, or it is delayed, or the namespace tree it reports does not terminate.
    ///
    /// The failing arms prove the failure policy (a namespace whose children or tables cannot be
    /// listed fails construction loudly instead of vanishing). The delay arm makes provider
    /// construction complete OUT of request order, which is what makes the ordered-`buffered`
    /// requirement observable. The [`ParentScript`] arms prove the `seen` visited-set is what
    /// terminates a walk whose depth never grows.
    #[derive(Debug)]
    struct ScriptedCatalog {
        inner: Arc<MemoryCatalog>,
        fail_children_of: Option<NamespaceIdent>,
        fail_tables_of: Option<NamespaceIdent>,
        delay_tables_of: Option<NamespaceIdent>,
        parent_script: ParentScript,
        /// Every `list_namespaces` call, so a test can pin the N+1 listing shape and the budget
        /// can stop a non-terminating walk.
        listing_calls: AtomicUsize,
    }

    impl ScriptedCatalog {
        fn delegating(inner: Arc<MemoryCatalog>) -> Self {
            Self {
                inner,
                fail_children_of: None,
                fail_tables_of: None,
                delay_tables_of: None,
                parent_script: ParentScript::Delegate,
                listing_calls: AtomicUsize::new(0),
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
    }

    #[async_trait]
    impl Catalog for ScriptedCatalog {
        async fn list_namespaces(
            &self,
            parent: Option<&NamespaceIdent>,
        ) -> Result<Vec<NamespaceIdent>> {
            let calls = self.listing_calls.fetch_add(1, Ordering::Relaxed) + 1;
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
            // Sorted so the REQUEST order of provider construction is deterministic. Without this
            // the memory catalog yields namespaces in hash order, and
            // `providers_stay_bound_to_their_own_namespace_when_listings_finish_out_of_order`
            // would only sometimes be able to distinguish `buffered` from `buffer_unordered`.
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
            if self.fail_tables_of.as_ref() == Some(namespace) {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    "simulated table-listing failure",
                ));
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

    /// The single-level shape that existed before nested discovery: unchanged behaviour, one
    /// schema per namespace, named exactly as before.
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

    /// OTH-001 (a): the multi-level namespace `["a", "b"]` must survive as ONE schema carrying its
    /// own tables — not as two bare schemas `a` and `b`.
    ///
    /// Mutation this catches: restoring the original
    /// `.flat_map(|ns| ns.as_ref().clone())` + `NamespaceIdent::new(name)` explosion — the bare
    /// `"b"` assertion goes RED (a non-existent schema appears) and the canonical `a\u{1f}b`
    /// lookup goes RED (the real namespace is gone).
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

        // The exploded component must NOT exist as a schema of its own.
        assert!(
            provider.schema("b").is_none(),
            "the bare level \"b\" was registered as a schema; the namespace was flattened"
        );
    }

    /// OTH-001 (b): discovery recurses, and it recurses past two levels.
    ///
    /// Mutation this catches: deleting the `frontier = list_child_namespaces(..)` line (or the
    /// whole BFS loop body) so only `list_namespaces(None)` is walked — the three-level canonical
    /// lookup goes RED.
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

    /// The canonical name round-trips back to the exact namespace, which is the property the
    /// separator choice is documented on.
    ///
    /// Mutation this catches: changing `canonical_schema_name` to join with `.` — the recovered
    /// identifier for `["a.b", "c"]` becomes `["a", "b", "c"]` and the assertion goes RED.
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

    /// The SQL-typeable alias resolves a nested namespace to the same provider as its canonical
    /// name.
    ///
    /// Mutation this catches: deleting the `aliases` lookup arm from
    /// `CatalogProvider::schema` — `schema("x.y")` becomes `None` and the test goes RED.
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

        // The alias is resolution-only: it must not appear in the listed schema names.
        assert!(
            !provider.schema_names().contains(&"x.y".to_string()),
            "the alias leaked into schema_names()"
        );
    }

    /// COLLISION 1 — `["a", "b"]`'s alias `a.b` is also `["a.b"]`'s CANONICAL name. The canonical
    /// binding must win, and the aliased namespace must stay reachable under its own canonical
    /// name. Neither may be served in place of the other.
    ///
    /// Mutation this catches: dropping the `schemas.contains_key(&alias)` guard so the alias is
    /// inserted anyway. The alias map is consulted only after the canonical map, so the wrong
    /// provider would not be returned by `schema("a.b")` — instead the guard's absence is caught
    /// by the assertion that the alias map has no entry that shadows a canonical name, which the
    /// two table-name assertions below pin end to end.
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

        // `a.b` is the single-level namespace's own name — it must serve that namespace's table.
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

    /// COLLISION 2 — two DIFFERENT nested namespaces claim the same alias `a.b.c`. Neither may
    /// win: the alias must resolve to nothing so a query fails loudly rather than reading the
    /// wrong namespace's tables.
    ///
    /// Mutation this catches: removing the `alias_claims.get(&alias) > 1` guard — the alias then
    /// resolves to whichever namespace was inserted last and `schema("a.b.c").is_none()` goes RED.
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

        // Both namespaces remain reachable under their unambiguous canonical names.
        assert!(provider.schema(&canonical(&["a", "b", "c"])).is_some());
        assert!(provider.schema(&canonical(&["a", "b.c"])).is_some());
    }

    /// COLLISION 3 — a namespace level containing the canonical separator would make the join
    /// non-injective (`["a\u{1f}b"]` vs `["a", "b"]`). Construction must fail with a typed error
    /// naming the namespace, not silently keep one of them.
    ///
    /// Mutation this catches: deleting the `level.contains(NAMESPACE_SEPARATOR)` arm of
    /// `validate_namespace_renderable` — `try_new` then returns `Ok` with one namespace silently
    /// overwriting the other, and `is_err()` goes RED.
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
    /// Mutation this catches: changing the guard to `namespace.len() > MAX_NAMESPACE_DEPTH + 1`
    /// (or deleting it) — the over-deep namespace is accepted and `expect_err` goes RED.
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

    /// Builds a two-root catalog whose child listings are scripted by `parent_script`, then runs
    /// discovery under a wall-clock timeout so a non-terminating walk FAILS instead of wedging the
    /// test binary. Returns the provider, the scripted catalog (so the caller can pin how many
    /// listings the walk cost) and the warehouse dir, which must outlive both.
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

    /// TERMINATION 1 — a server that ignores the parent filter and answers every child listing
    /// with the ROOT list. This is exactly what a REST catalog does when it drops an unrecognised
    /// `?parent=` query parameter, and the namespaces it re-reports are already visited, so the
    /// walk's depth NEVER grows and [`MAX_NAMESPACE_DEPTH`] is never reached. The `seen`
    /// visited-set is the only thing that stops it.
    ///
    /// Mutation this catches: replacing
    /// `if seen.insert(namespace.clone()) { discovered.push(..); fresh.push(..); }` in
    /// [`discover_namespaces`] with the unconditional `discovered.push(..); fresh.push(..);`
    /// (equivalently: deleting `seen`). The frontier then never empties, the walk re-lists the same
    /// two namespaces forever, [`MAX_SCRIPTED_LISTINGS`] trips, and construction returns `Err` —
    /// RED. Against a real network catalog the same mutant is an unbounded request flood at
    /// session construction.
    #[tokio::test]
    async fn a_catalog_that_ignores_the_parent_filter_still_terminates() {
        let (provider, scripted, _warehouse) =
            discover_under_script(ParentScript::IgnoreParent).await;

        assert_eq!(sorted_schema_names(&provider), vec![
            "ns1".to_string(),
            "ns2".to_string()
        ]);
        // One root listing plus one child listing per namespace — the N+1 shape, walked ONCE.
        assert_eq!(
            scripted.listings_issued(),
            3,
            "the walk re-expanded an already-visited namespace"
        );
    }

    /// TERMINATION 2 — a genuine cycle: the catalog reports `ns1`'s child as `ns2` and `ns2`'s
    /// child as `ns1`. Depth never increases here either, so again only `seen` terminates the walk,
    /// and both namespaces must still be discovered exactly once.
    ///
    /// Mutation this catches: the same `seen.insert` removal as
    /// `a_catalog_that_ignores_the_parent_filter_still_terminates` — the ring is walked forever,
    /// the listing budget trips and construction returns `Err`.
    ///
    /// Second mutation, applied and confirmed RED: moving `let mut seen = HashSet::new()` INSIDE
    /// the `while` loop, so de-duplication is per-round rather than global. The frontier here is
    /// one namespace wide at every round and never repeats WITHIN a round, so termination depends
    /// on the visited-set surviving ACROSS rounds — which is the property the declaration site
    /// carries. (That mutant is RED for the parent-ignoring shape too; the two tests are not
    /// distinguished by it, they are distinguished by frontier width and by whether the repeat is
    /// intra- or inter-round.)
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

    /// A namespace whose CHILD listing fails must fail construction and name itself, never be
    /// dropped from the catalog.
    ///
    /// Mutation this catches: swallowing the error in `list_child_namespaces` (e.g.
    /// `.unwrap_or_default()`) — `expect_err` goes RED.
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

    /// A namespace whose TABLE listing fails must likewise fail construction and name itself.
    ///
    /// Mutation this catches: replacing `try_collect` in `build_schema_providers` with a
    /// filter-out-the-errors collect — `expect_err` goes RED.
    #[tokio::test]
    async fn a_table_listing_failure_fails_construction_and_names_the_namespace() {
        let (inner, _dir) = memory_catalog().await;
        create_namespace_chain(&inner, &["good"]).await;
        create_namespace_chain(&inner, &["broken", "leaf"]).await;
        let broken = NamespaceIdent::from_strs(["broken", "leaf"]).expect("bad namespace");

        let catalog: Arc<dyn Catalog> = Arc::new(ScriptedCatalog::failing_tables_of(inner, broken));

        let err = IcebergCatalogProvider::try_new(catalog)
            .await
            .expect_err("a failing table listing must fail construction");

        assert!(
            err.to_string().contains("schema provider")
                && err.to_string().contains("\"broken\", \"leaf\""),
            "the error does not name the namespace whose provider failed: {err}"
        );
        assert!(
            std::error::Error::source(&err).is_some(),
            "the error chain to the underlying listing failure was broken"
        );
    }

    /// Providers must stay aligned with the namespaces they are zipped against even when the
    /// catalog answers out of request order — otherwise a query on one schema silently reads
    /// another namespace's tables.
    ///
    /// [`ScriptedCatalog`] lists namespaces sorted, so `a_delayed` is requested FIRST; its table
    /// listing then sleeps, so it completes LAST. Completion order is therefore the reverse of
    /// request order — deterministically, not by luck.
    ///
    /// Mutation this catches: `buffered` → `buffer_unordered` in `build_schema_providers` — the
    /// completion-ordered results zip onto the wrong names and both table assertions go RED.
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
}
