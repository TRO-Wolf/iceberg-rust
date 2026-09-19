# F-CATALOG-CACHE-1 — shared metadata and manifest caches for S3 Tables and Glue

Unit ledger. Scope: one bounded, scoped, concurrent `TableMetadataCache` shared by
catalog handles, plus a per-catalog-instance `ObjectCache` for manifests, wired into
`S3TablesCatalogBuilder` and `GlueCatalogBuilder` with the memory catalog's existing
opt-in semantics. The memory catalog adopts the scoped key.

Status: implemented. All pins green; mutation evidence below; gates green.

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

## Offline seam (as built)

Each service catalog gets a `#[cfg(test)]` pointer source — a shared
`Arc<dyn Fn(&TableIdent) -> Result<(location, version)>>` consulted at the top of
`get_table_pointer`, before the outcome-harness arm and the SDK call
(`crates/catalog/s3tables/src/catalog.rs:359`,
`crates/catalog/glue/src/catalog/caches.rs:115`). This runs (a)+(b) through the real
production path — unlike the outcome harness, which returns a table and bypasses
the cache. `with_pointer_source` and `with_file_io_for_tests` are the test seams.

Deviation from the planned `CountingStorage` wrapper: neither catalog crate has the
serde/typetag deps a custom `Storage` impl needs (Cargo.toml is owner-gated), so
bodies live in `FileIO::new_with_memory()` and observability comes from
`cache.stats()` (`body_fetches` counts exactly one increment per metadata body
parse), `Arc::ptr_eq` on `metadata_ref()`, metadata-body rewrites under the same
location, and `ObjectCache` pointer identity. The same claims are pinned with the
same strength.

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

Shared cache — `crates/iceberg/src/catalog/table_metadata_cache.rs`:

- D-2 `CacheScope` (catalog_identity + credential_context): lines 57–102;
  `CREDENTIAL_CONTEXT_PROP_KEYS` line 32 (non-secret selectors only);
  `for_catalog` falls back to `instance:<uuid>` when no context derivable (93–96).
- D-3 bound: `DEFAULT_MAX_ENTRIES = 1024` line 30; moka `max_capacity` +
  `RemovalCause::Size` eviction counter 154–166; `stats().evictions`.
- D-4 dedup: `try_get_with` single-flight in `load_or_fetch_table_metadata`
  307–311; init error returned to every waiter as `Arc<Error>` and never cached
  (317–321).
- Fail-closed version guard: `version_conflicts` 117–122; `lookup` 240–253;
  `arm_version` 255–263 (unversioned cached entry learns a supplied token).
- Key: `CacheKey { scope, location }` 104–108; location is always required.
- `load_or_fetch_table_metadata` 279–336: no cache handle → body read every
  load (286–288); version-conflict refetch loop 297–322; failed refetch leaves
  stale state invalidated (298).
- Tests: `crates/iceberg/src/catalog/table_metadata_cache_tests.rs`
  (`#[cfg(test)] mod tests { include!(...) }` at table_metadata_cache.rs:339;
  split required by the 1000-line default ceiling).

Memory catalog — `crates/iceberg/src/catalog/memory/`:

- Scope `memory:<warehouse>` + prop-derived context: catalog.rs:171;
  `with_cache_credential_context` in memory/caches.rs:34.
- Load path through `load_or_fetch_table_metadata`: catalog.rs:213, 500.
- D-6 seeding after create/register: 429; after update CAS: 616. Invalidations:
  register-failure 459, missing-table 516, update old-pointer 614, drop 632.
  Mutex guards are dropped before every `.await`ed cache call.
- `Table::object_cache()` promoted `pub(crate)` → `pub`
  (`crates/iceberg/src/table.rs:249`) so external catalog tests can assert
  per-catalog `ObjectCache` identity (P-7).

S3 Tables — `crates/catalog/s3tables/`:

- Builder handles + scope: builder fields catalog.rs (`table_metadata_cache`,
  `shared_object_cache_bytes`, `cache_credential_context`), applied via
  `with_cache_options` catalog.rs:208; scope `s3tables:<table_bucket_arn>`
  catalog.rs:270.
- D-1 pointer-first load: `get_table_pointer` catalog.rs:350 (pointer_source
  seam 359); `load_table_with_version_token` catalog.rs:415–424 — pointer, then
  `load_or_fetch_table_metadata` with `Some(&version_token)`.
- `resolve_commit_base` FullLoad leg: catalog.rs:475.
- D-5 `table_builder()` shares the catalog `ObjectCache` on every table:
  caches.rs `table_builder`; used at catalog.rs:395, 426, 462, 484, 715.
- D-6 seeding after landed writes: create catalog.rs:712; update
  catalog.rs:815; publish-replace catalog.rs:860.
- Tests: `crates/catalog/s3tables/src/cache_tests.rs` (P-1, P-2, P-4, P-6, P-7,
  P-8, P-9). Prior inline `mod tests` extracted to `catalog_tests.rs` via
  `include!` to keep catalog.rs under the legacy ceiling; the obsolete ceiling
  rows were then removed from `scripts/check_rust_file_size.py` per the
  checker's own contract (file is below the 1000 default).

Glue — `crates/catalog/glue/`:

- Builder handles: `GlueCatalogBuilder::{with_table_metadata_cache,
  with_shared_object_cache_bytes, with_cache_credential_context}` in
  `catalog/caches.rs`; applied via `with_cache_options` catalog.rs:157.
- Scope `glue:<catalog_id|default>:<region_name|default>:<warehouse>`:
  catalog.rs:289–300.
- Load path lives in `catalog/caches.rs` (the load path IS the cache path —
  moved to keep catalog.rs under its legacy ceiling; comments died in the
  move per Rule 0): `get_table_pointer` caches.rs:104 (pointer_source seam
  115); `load_table_with_version_id` caches.rs:147–202 — pointer, then
  `load_or_fetch_table_metadata` with `version_id.as_deref()`;
  `resolve_commit_base` caches.rs:204–268 (FullLoad leg 245–254).
- `register_table` reads through `load_or_fetch_table_metadata`
  catalog.rs:815 (no version known → unarmed guard).
- D-5 `table_builder()`: caches.rs:64; used at caches.rs:157, 189, 229, 256
  and catalog.rs:630, 860.
- D-6 seeding after landed writes: create catalog.rs:628; register (the
  load_or_fetch read itself installs the entry) catalog.rs:815; update
  catalog.rs:909; publish-replace `catalog/replace_publish.rs:100–102`.
- Tests: `crates/catalog/glue/src/catalog/cache_tests.rs` (P-1, P-2, P-4,
  P-6, P-7, P-8, P-9).

## Mutation results (all mutations applied, observed red, reverted)

- M1 pointer memoization by table name (`static POINTER_MEMO:
  Mutex<BTreeMap>` consulted before `get_table_pointer`):
  s3tables — `p1_second_handle_commit_visible_on_first_handle_next_load` RED
  (`v1.metadata.json` served after pointer moved to `v2`) and
  `p2_external_pointer_move_visible_on_next_load` RED (v1 served, expected
  v9). Glue — same two pins RED with the identical mutation. Proves P-1/P-2
  are load-bearing on the pointer being re-fetched every load.
- M2 `key()` ignores `scope` (constant `CacheScope::new("global","global")`):
  `different_scopes_same_location_never_share` RED — body_fetches 1, expected
  2. Proves P-4 pins the scope in the key.
- M3 `try_get_with` replaced by fetch+insert (no single-flight):
  `concurrent_cold_loads_dedup_single_fetch_and_errors_reach_all` RED —
  body_fetches 16, expected 1. Proves P-5 pins dedup.
- M4 `max_capacity(u64::MAX)` ignoring the configured bound:
  `eviction_under_pressure_bounds_and_counts_and_never_stale` RED on the
  bound assertion (`cache must hold at most the configured bound: 5`) — and
  reached line 549, i.e. every staleness assertion before it stayed GREEN.
  Keeping evicted entries can never serve stale data because the key is an
  immutable metadata location reached through the fresh pointer (D-1);
  eviction is a capacity concern, not a correctness one. The red leg is the
  bound, exactly as predicted.

## Gates

- `cargo fmt --all -- --check` — clean.
- `cargo clippy -p iceberg -p iceberg-catalog-s3tables -p iceberg-catalog-glue
  --all-targets -- -D warnings` — clean.
- `./scripts/check_rust_file_size.sh` — clean (obsolete glue + s3tables
  ceiling rows removed per the checker's contract; both files under the 1000
  default). `python3 -B -m unittest scripts/check_rust_file_size_test.py` —
  11 tests OK.
- `./scripts/check_comment_blocks.sh` — clean.
- `./scripts/check_agent_artifacts.sh` — clean.
- `./scripts/check_matrix_anchors.sh` — clean (88 rows).
- `taplo check` — clean. `cargo machete` — clean. `typos` — clean.
- `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/pb-fork2 origin/main HEAD`
  — `comment-ban hits=0` after every commit.
- `cargo test -p iceberg --lib table_metadata_cache` — 12/12.
- `cargo test -p iceberg --lib catalog::memory` — 95/95.
- `cargo test -p iceberg-catalog-s3tables --lib` — 46/46.
- `cargo test -p iceberg-catalog-glue --lib` — 57/57.
