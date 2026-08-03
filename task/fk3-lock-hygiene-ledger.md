# FK3 — lock hygiene pair (scout #12 + #13)

**Branch:** `feat/fk-mor-perf-campaign`  
**Tag:** `[fork]`  
**Base (campaign):** `a966055e` (#182)  
**FK2.1 tip before this unit:** `af1a4f28` (OCTO-CONVERGED)  
**Worktree:** `/tmp/iceberg-rust-fk3`

## Sequencing
FK3 is one unit, two cuts (campaign brief):
1. **#12** freeze pos-delete vectors without `Mutex` after publish
2. **#13** MemoryCatalog: do not hold global mutex across FileIO

Do **not** start FK4 from this unit.

---

## #12 — freeze `Arc<DeleteVector>` (audit first)

### Audit finding
**Scout premise CONFIRMED.** Post-publish mutation of a memoized positional delete
vector does **not** exist on the load → install → resolve → apply path:

| Stage | Mutation? | Notes |
|---|---|---|
| `install_pos_del_contribution` | once, under write lock | installs `Arc<HashMap<path, DeleteVector>>`; never re-inserted for same claim |
| `resolve_delete_vector` merge | builds a **new** `DeleteVector` | OR-by-ref via `merge(&contrib)`; memoized as `Arc` |
| memo hit / apply / reader | **read-only** | `contains` / `iter` / `is_empty` / range-walk keep-mask |
| public `deleted_row_positions` | **read-only** | same Arc |

No path calls `insert` / `merge` / `|=` on a vector already published into
`resolved_pos_dels`. Contribution maps are also immutable once installed (comments
pre-FK3 already stated this; audit agrees). **Therefore freeze is safe** — keep lock
was the wrong choice.

### Change
- `resolved_pos_dels: HashMap<…, Arc<DeleteVector>>` (was `Arc<Mutex<DeleteVector>>`)
- Merge: `merged.merge(vector)` by reference — no roaring full-clone before OR
- `get_delete_vector` / `resolve_delete_vector` / `deleted_row_positions` return
  `Option<Arc<DeleteVector>>`
- `apply` + parquet/avro/orc survival paths: lock-free on the bitmap
- Call sites updated: `reader.rs`, loader tests, interop_scan_exec, row_delta helper,
  DV writer tests

### Public API disclosure
| API | Before | After |
|---|---|---|
| `DeleteFilter::deleted_row_positions` | `Option<Arc<Mutex<DeleteVector>>>` | `Option<Arc<DeleteVector>>` |

Engines that locked the mutex must drop the lock (immutable Arc). Serde N/A.

### Pins
- `test_resolved_pos_del_vector_is_frozen_arc_shared` — `Arc::ptr_eq` on double resolve
- Existing poison-recovery pin still covers the **outer** state `RwLock` (not a
  per-vector mutex)
- Full delete_filter + caching_delete_file_loader suites green

---

## #13 — MemoryCatalog short critical sections + optimistic CAS

### Change
FileIO **outside** the catalog mutex; pointer ops only under short sections:

| Op | Pattern |
|---|---|
| `load_table` / `load_view` | snapshot location under lock → read FileIO outside |
| `create_table` / `create_view` | resolve location under lock → write FileIO outside → insert under lock |
| `register_table` | **read FileIO first** (fail closed) → insert under lock |
| `drop_table` | remove pointer under lock → delete file outside |
| `update_table` / `update_view` | snapshot → load/apply/write outside → **re-read + CAS + flip** under lock |

Authoritative CAS is always the flip-time re-read (early CAS against the load snapshot
is a cheap pre-filter). Concurrent winners advance the stored location → retryable
`CatalogCommitConflicts` — same semantics as holding the lock for the whole body.

### Atomicity pins
- `test_register_table_unreachable_metadata_refuses_half_create` — failed read leaves
  `table_exists == false`; subsequent create of same ident succeeds
- `test_table_stale_commit_conflicts_with_io_outside_lock` — O1 CAS still fires under
  short-critical-section shape
- Existing view/table stale-CAS + refresh pins remain green

### Latency note (structural; no microbench wall-time histogram)
**Before:** one global `Mutex` held across metadata `read_from` / `write_to` for load,
register, create, update — concurrent sessions serialized on FileIO.  
**After:** only pointer snapshot / insert / CAS flip hold the mutex; FileIO proceeds
concurrently. Pin `test_concurrent_load_during_update_completes` asserts liveness of
parallel load+update (not a ns histogram).

---

## Mid-unit gate

| Gate | Command | Exit |
|---|---|---|
| delete_filter | `cargo test -p iceberg --lib delete_filter` | **0** (41 passed) |
| caching_delete_file_loader | `cargo test -p iceberg --lib caching_delete_file_loader` | **0** (35 passed) |
| memory catalog | `cargo test -p iceberg --lib catalog::memory` | **0** (incl. new FK3 pins) |
| clippy lib | `cargo clippy -p iceberg --lib -- -D warnings` | **0** |

## map.md
No `map.md` under `arrow/` or `catalog/` (convention not present) — no map update.

## Not in this unit
- FK4 (#7 metadata-pointer cache, #30 OpenDAL list)
- Cargo.toml (frozen)
- GAP_MATRIX (perf only — no parity claim)

## Critic-octo
Scratch: `/tmp/critic-octo-fk3-2026-08-08/`  
**Label:** **OCTO-CONVERGED** (8/8, `early_stop=false`)  
**Actor tip:** `5e0181ed`  
**Critic tip:** `b04b3c38`

### Critic fixes (by cycle)
| Cycle | Finding | Fix |
|------:|---|---|
| 1 | S2 multi-source OR unpinned; S2 orphan before flip | `test_multi_source_resolve_ors_by_ref_into_frozen_arc`; mid-point CAS recheck before write (table+view) |
| 1 | S3 trait atomicity prose | `Catalog::publish_create_table` note: FileIO may leave the lock |
| 2–3 | create orphan-on-lose; concurrent load isolation | WITHDRAWN intentional / pre-existing |
| 4–8 | API disclosure / concurrency / reader / view / gate | ledger + CLEAN re-proof |

### Mutation RED
| Gate | Break | Pin turns RED |
|---|---|---|
| Freeze Arc share | fresh `Arc` per resolve / no memo | `test_resolved_pos_del_vector_is_frozen_arc_shared` |
| Multi-source OR | merge only first contribution | `test_multi_source_resolve_ors_by_ref_into_frozen_arc` |
| Half-create refuse | insert pointer before FileIO read | `test_register_table_unreachable_metadata_refuses_half_create` |
| Stale CAS | skip flip-time location compare | `test_table_stale_*` / view stale pin |

### Residual OPEN (≥ S1: **none**)
- **S3 seed:** residual orphan metadata file if a concurrent winner lands between mid-point recheck and flip CAS (pointer never half-flips)
- **S3 seed:** create_table loser after write leaves orphan metadata (pre-existing shape under full-lock write-then-insert too)
