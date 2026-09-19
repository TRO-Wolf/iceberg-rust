# F-CATALOG-CACHE-1 — shared metadata and manifest caches for S3 Tables and Glue

Unit ledger. Scope: one bounded, scoped, concurrent `TableMetadataCache` shared by
catalog handles, plus a per-catalog-instance `ObjectCache` for manifests, wired into
`S3TablesCatalogBuilder` and `GlueCatalogBuilder` with the memory catalog's existing
opt-in semantics. The memory catalog adopts the scoped key.

Status: in progress (design clauses below; evidence lines filled as slices land).

## Orchestrator rulings restated

### D-1 — pointer check on every load

`load_table` always fetches the service pointer first, exactly as today:

- S3 Tables: `S3TablesCatalog::get_table_pointer` — `GetTable` → `metadata_location` +
  `version_token` (`crates/catalog/s3tables/src/catalog.rs`, `get_table_pointer`).
- Glue: `GlueCatalog::get_table_pointer` — `GetTable` → `metadata_location` +
  `VersionId` (`crates/catalog/glue/src/catalog.rs`, `get_table_pointer`).
- Memory: `MemoryCatalog::table_metadata_location` — in-process `NamespaceState`
  pointer, no service version (`crates/iceberg/src/catalog/memory/catalog.rs`).

The metadata cache is consulted only AFTER the pointer, keyed by the pointer's
location, with the service version as the fail-closed object-version guard the cache
already supports. The cache never answers "where is the current metadata"; external
writers and service-side compaction stay visible on the next load.

Evidence: `load_table_with_version_token` /
`load_table_with_version_id` split into (a) `get_table_pointer` and (b)
`load_table_from_pointer(ident, location, version)` per catalog. Pins: P-1, P-2.

### D-2 — scoped keys

The cache key becomes `(CacheScope, metadata_location)`. `CacheScope` is a public
type in `table_metadata_cache.rs` carrying two non-secret strings:

- `catalog_identity`: S3 Tables `s3tables:<table_bucket_arn>`; Glue
  `glue:<catalog_id or "default">:<region or "default">:<warehouse>`; memory
  `memory:<warehouse>`.
- `credential_context`: the non-secret identifiers that select the credentials in
  the catalog's props — access key id, profile name, assume-role ARN / session name,
  region. Never a secret key or session token; never logged. Derived by
  `CacheScope::credential_context_from_props` from a fixed non-secret key list
  (`aws_access_key_id`, `profile_name`, `region_name`, `s3.access-key-id`,
  `s3.region`, `client.assume-role.arn`, `client.assume-role.session-name`).

Fail closed when no credential context can be established — injected SDK client
(s3tables `with_client`, credentials unknown) or no credential props (default chain):
the scope carries a per-catalog-instance `instance:<uuid>` so nothing is shared
across instances. `with_cache_credential_context(String)` on each builder names the
context explicitly and overrides derivation.

Two scopes never share an entry, even for the same location string. Pins: P-4
(cache-level two scopes, catalog-level two bucket ARNs / two Glue catalog ids /
two credential contexts).

### D-3 — bounded

`TableMetadataCache::with_max_entries(n)` sets the bound. `new()` keeps today's
semantics for existing callers but is bounded: **default bound = 1024 entries**.

Rationale for 1024: each entry is one parsed `TableMetadata` Arc (tens of KB typical);
a session touching more than ~1024 distinct metadata locations (hundreds of tables ×
a few commits) is already an outlier, and the RePark consumer previously cleared the
whole map when `len() > entries` — a bound is strictly better. 1024 bounds a session
to order-of-magnitude ~50–100 MB worst-case metadata retention, and eviction can
only ever cost one refetch because the key is an immutable metadata location reached
through D-1 (never a stale answer).

Eviction is via `moka::future::Cache` (`max_capacity`, TinyLFU admission); stats gain
`evictions` counting `RemovalCause::Size` events via the eviction listener. Pin: P-3.

### D-4 — concurrent-miss deduplication

N concurrent loads of one key with a cold cache perform exactly one body GET and
parse: moka `try_get_with` shares the in-flight init across waiters on the same
`(scope, location)` key. An init error is delivered to every waiter (as a
reconstructed `Error` preserving kind/message/retryable — `iceberg::Error` is not
`Clone`, moka hands every waiter `Arc<E>`) and is NOT cached, so a later load
retries. Pin: P-5 (N=16).

### D-5 — same two handles as memory

`S3TablesCatalogBuilder` and `GlueCatalogBuilder` gain
`with_table_metadata_cache(Arc<TableMetadataCache>)` and
`with_shared_object_cache_bytes(u64)` with the memory catalog's semantics. The
object cache is one `ObjectCache` per catalog instance built over that catalog's own
`FileIO` (`ObjectCache::new_with_capacity(file_io, bytes)`, `bytes == 0` → OFF) and
every `Table` the catalog builds (load, create, register, commit, publish-replace)
shares it via `TableBuilder::object_cache`. Both handles default OFF. The memory
catalog adopts the scoped key; `MemoryCatalogBuilder` gains
`with_cache_credential_context` for symmetry. Pins: P-6 (warm reload), P-7 (object
cache per instance, manifest read once), P-9 (defaults off).

### D-6 — writes publish

After a successful commit / create / register, the catalog puts the metadata it just
wrote under `(scope, new location)` — never under an old location:

- s3tables: `create_table` (after `UpdateTableMetadataLocation` lands),
  `update_table`, `publish_replace_table` (after the CAS lands).
- glue: `create_table`, `register_table`, `update_table`, `publish_replace_table`.
- memory: existing `cache_put` sites unchanged in behaviour, scoped key adopted.

## Offline seam

Each service catalog gets a `#[cfg(test)]` pointer source: a shareable
`HashMap<TableIdent, (location, version)>` consulted in `get_table_pointer` after
the outcome-harness arm and before the SDK call, and advanced on a landed commit via
a hook next to the existing `publish_outcome_harness` call. This runs (a)+(b)
through the real production path — unlike the outcome harness, which returns a table
and bypasses the cache. (b) `load_table_from_pointer` is pinned directly.

Object bodies live in `MemoryStorage` behind a counting `Storage`/`StorageFactory`
wrapper per catalog crate (modelled on `CountingStorage` in
`table_metadata_cache` tests), counting `read` per path.

## Pins

| Pin | Claim | Where |
|---|---|---|
| P-1 | second handle's commit is visible on the first handle's next load | `catalog/cache_tests.rs` per crate |
| P-2 | external pointer move visible on next load | same |
| P-3 | eviction under pressure never returns stale; evictions counted; bound holds | `table_metadata_cache.rs` tests |
| P-4 | two scopes with the same location string never share an entry | cache-level + catalog-level per crate |
| P-5 | 16 concurrent cold loads → 1 body fetch; error reaches all 16; retry succeeds | `table_metadata_cache.rs` tests |
| P-6 | warm reload, unchanged pointer: zero body GETs, same `Arc` | `catalog/cache_tests.rs` per crate |
| P-7 | object cache: two loads + scan plan fetch each manifest once; two catalog instances never share an `ObjectCache` | `catalog/cache_tests.rs` per crate |
| P-8 | same location, different service version → refetch (fail closed) | `table_metadata_cache.rs` tests + glue leg |
| P-9 | defaults OFF: body GET on every load | `catalog/cache_tests.rs` per crate |

## Mutation plan

- M1 pointer staleness: make the test pointer source sticky-first (`set`/`publish`
  no-op once seeded) → P-1 and P-2 must go red.
- M2 scope drop: `key()` ignores `scope` → P-4 must go red.
- M3 dedup removal: `try_get_with` replaced by get+fetch+insert → P-5 must go red.
- M4 unbounded: `max_capacity` ignored → P-3 bound assertion red; the
  stale-metadata leg stays green (keys are immutable locations — explain).

## Evidence (filled per slice)

TBD.

## Gates

TBD.
