# FK4 — I/O pair (scout #7 + #30)

**Branch:** `feat/fk-mor-perf-campaign`  
**Tag:** `[fork]`  
**Base (campaign):** `a966055e` (#182)  
**FK3 tip before this unit:** `fa067b48` (OCTO-CONVERGED)  
**Worktree:** `/tmp/iceberg-rust-fk4_1`

## Sequencing

Campaign brief: **FK4.1** (scout #7 metadata-pointer cache) **before** FK4.2 (scout #30 OpenDAL list).

| Unit | Status | Notes |
|---|---|---|
| **FK4.1** metadata-pointer cache | **THIS UNIT** | opt-in session cache at catalog `load_table` |
| **FK4.2** OpenDAL list concurrent stat | **NOT STARTED** | do not begin from this unit |

Also do **not** start FK2.2 / FK2.3 from this unit.

---

## FK4.1 — metadata-pointer cache (scout #7)

### Design (locked)

| Rule | Implementation |
|---|---|
| Optional inject at catalog `load_table` path (b+c) | `MemoryCatalogBuilder::with_table_metadata_cache(Arc<TableMetadataCache>)` + `load_or_fetch_table_metadata` on load/register |
| Opt-in, **default OFF** | `Option<Arc<TableMetadataCache>> = None` when builder omits injection |
| No global / thread-local state | Caller-owned `Arc` only |
| Fail CLOSED on any mismatch | Full body GET + re-parse; never soft-reuse |
| v1 key = metadata-location **string equality** | `HashMap<String, CachedEntry>` |
| ETag / version-id free extra guard | `lookup(location, Option<&str>)` — checked only when **both** sides `Some`; never sole check |
| Bench / pin | op-count injector (`CountingStorage` body `read`s) + cache `body_fetches` / hits |

Scout soft-rebase on UUID data paths is **OUT** for v1.

### API surface (opt-in)

```text
// crates/iceberg (re-exported from catalog)
pub struct TableMetadataCache { … }
pub struct TableMetadataCacheStats { hits, misses, body_fetches }
impl TableMetadataCache {
    pub fn new() -> Self
    pub fn stats(&self) -> TableMetadataCacheStats
    pub fn reset_stats(&self)
    pub fn len(&self) / is_empty(&self)
    pub fn put(location, TableMetadataRef, Option<String> /* object_version */)
    pub fn lookup(location, Option<&str> /* object_version */) -> Option<TableMetadataRef>
    pub fn invalidate(location)
    pub fn clear()
}
pub async fn load_or_fetch_table_metadata(
    file_io: &FileIO,
    metadata_location: &str,
    cache: Option<&TableMetadataCache>,
    object_version: Option<&str>,
) -> Result<TableMetadataRef>

// Memory catalog injection
MemoryCatalogBuilder::with_table_metadata_cache(Arc<TableMetadataCache>) -> Self
// Catalog::invalidate_table — when cache injected, evicts that table's location
//   (clear-all if location cannot be resolved — fail closed)
```

Other catalogs can call `load_or_fetch_table_metadata` the same way; **only MemoryCatalog is wired in this unit** (test vehicle + default-OFF baseline). Glue/S3 Tables adopt in a later polish if needed — helper is shared.

### Files

| Path | Change |
|---|---|
| `crates/iceberg/src/catalog/table_metadata_cache.rs` | **NEW** — cache + helper + unit pins (op-count injector) |
| `crates/iceberg/src/catalog/mod.rs` | module + re-exports |
| `crates/iceberg/src/catalog/memory/catalog.rs` | builder inject; load/register via helper; create/update seed; `invalidate_table` |
| `task/fk4-io-pair-ledger.md` | this ledger |

### Pins

| Pin | Claim |
|---|---|
| `two_loads_unchanged_pointer_zero_body_get_on_second` | storage body `read` count flat on 2nd load; `Arc::ptr_eq` |
| `default_off_always_body_gets` | `cache=None` → 2 loads = 2 body GETs |
| `object_version_mismatch_fail_closed_refetches` | guard disagree → re-GET |
| `location_change_is_miss` | new pointer key → miss |
| `invalidate_forces_refetch` | invalidate → next load body-GET |
| `version_never_sole_check_location_required` | wrong location + right etag → miss |
| `test_fk4_1_two_loads_unchanged_pointer_zero_body_fetch` | MemoryCatalog: create seed + 2 loads → 0 `body_fetches`, 2 hits, Arc share |
| `test_fk4_1_default_off_loads_without_cache` | no inject still loads |
| `test_fk4_1_pointer_change_on_update_is_new_key` | commit new location; load sees new props |
| `test_fk4_1_invalidate_table_evicts_pointer_entry` | `invalidate_table` → re-GET |
| `test_fk4_1_reload_same_pointer_is_cache_hit_commit_retry_leg` | commit-retry refresh of unchanged pointer = hit |

### Commit-retry note (cheap)

`Transaction` refresh on retryable conflict calls `catalog.load_table`. With the cache:

- **Unanged pointer** (refresh before a concurrent winner lands, or multi-statement load of the same base) → **cache hit, zero body GET** (pinned).
- **Winner advanced the pointer** → location string differs → **miss + full fetch** (correct fail-closed; no soft rebase).

No change to the retry loop itself in this unit.

### Mid-unit gate

| Gate | Command | Exit |
|---|---|---|
| cache + fk4_1 pins | `cargo test -p iceberg --lib -- table_metadata_cache fk4_1` | **0** (12 passed) |
| memory catalog | `cargo test -p iceberg --lib catalog::memory` | **0** (92 passed) |
| clippy lib | `cargo clippy -p iceberg --lib -- -D warnings` | **0** |

### map.md

No `map.md` under `catalog/` (convention not present) — no map update.

### Not in this unit

- **FK4.2** OpenDAL list concurrent stat
- **FK2.2 / FK2.3** plan-path cuts
- Cargo.toml (frozen)
- GAP_MATRIX (perf only — no parity claim)
- Soft rebase / UUID data-path reuse
- View metadata cache
- Wiring Glue / S3 Tables / REST / HMS / SQL builders (helper is ready; Memory is the vehicle)

### Critic-octo

Scratch: `/tmp/critic-octo-fk4_1-2026-08-08/`  
**Label:** **OCTO-CONVERGED** (8/8, `early_stop=false`)  
**Actor tip:** `52905e65`  
**Critic tip:** `bf652249`

### Critic fixes (by cycle)

| Cycle | Finding | Fix |
|------:|---|---|
| 1 | S2 missing invalidate cleared session | no-op on missing ident |
| 1 | S2 version guard never armed after seed | learn guard; mismatch fail-closes |
| 2 | S1 stale Arc after mismatch + failed GET | invalidate on miss before re-fetch |
| 3 | S2 drop left cache entry | drop_table invalidates |
| 4 | S2 seed before pointer claim | post-insert seed; register invalidate on insert Err |
| 5 | S2 prior pointer retained forever | evict prior on successful update |
| 6 | S2 eviction unpinned | `test_fk4_1_update_evicts_prior_pointer` |
| 7–8 | — | CLEAN re-proof |

### Mutation RED

| Gate | Break | Pin |
|---|---|---|
| Zero body GET | always fetch | op-count + `test_fk4_1_two_loads_*` |
| Guard mismatch | soft-reuse | `object_version_mismatch_*` / failed-refetch pin |
| Missing invalidate thrash | `clear()` | `test_fk4_1_invalidate_missing_*` |
| Drop / update eviction | skip invalidate | drop + update prior pins |

### Residual OPEN (≥ S1: **none**)

- **S3 seed:** no LRU / capacity bound
- **S3 seed:** only MemoryCatalog builder wired (helper shared)
- **S3 seed:** `publish_replace` does not seed (miss then fetch)
