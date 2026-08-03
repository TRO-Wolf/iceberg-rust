# FK2 — plan-path ownership + overlap (scout #5 + #14 + #15)

**Branch:** `feat/fk-mor-perf-campaign`  
**Tag:** `[fork]`  
**Base (campaign):** `a966055e` (#182)  
**FK1 tip before this unit:** `c7b14d80` (OCTO-CONVERGED)

## Sequencing
FK2 is three cuts; **strict serial** per campaign brief:
1. **FK2.1** Arc-share `FileScanTask` innards (scout #5) — done (mid-unit was 2.1)
2. **FK2.2** overlap delete/data manifest planning (scout #14) — **this unit** (after FK3 / FK4.1)
3. **FK2.3** delete-index keys (scout #15) — **this unit** (after FK3 / FK4.1)

---

## FK2.1 — Arc-share FileScanTask innards (scout #5)

### Hour-0 (optional)
Not measured this pass (plan-only heaptrack deferred). Structural win is unambiguous:
`sub_task` previously deep-cloned `String` path, `Vec<i32>` projection, `Option<BoundPredicate>`
residual tree, and `Vec<FileScanTaskDeleteFile>` deletes on every split window.

### Change
- `FileScanTask` shared fields are Arc-backed:
  - `data_file_path: Arc<str>`
  - `project_field_ids: Arc<[i32]>`
  - `predicate: Option<Arc<BoundPredicate>>`
  - `deletes: Arc<[FileScanTaskDeleteFile]>`
- Custom serde (`serde_arc_str` / `serde_arc_slice`) keeps JSON **byte-shape compatible**
  (path = JSON string; projection/deletes = JSON arrays; residual = bare `BoundPredicate`).
- `sub_task` Arc-clones shared pieces; only mutates `start` / `length` / `record_count` /
  `split_offsets`.
- `context.rs::residual_predicate` returns the memo `Arc` (no deep clone of the tree).
- `into_file_scan_task` builds Arc path/projection/deletes once per file.

### Public API disclosure
| Field | Before | After |
|---|---|---|
| `FileScanTask::data_file_path` | `String` | `Arc<str>` |
| `FileScanTask::project_field_ids` | `Vec<i32>` | `Arc<[i32]>` |
| `FileScanTask::predicate` | `Option<BoundPredicate>` | `Option<Arc<BoundPredicate>>` |
| `FileScanTask::deletes` | `Vec<FileScanTaskDeleteFile>` | `Arc<[FileScanTaskDeleteFile]>` |

Accessors (`data_file_path()`, `project_field_ids()`, `predicate()`) still return
`&str` / `&[i32]` / `Option<&BoundPredicate>`. Serde JSON shape unchanged (STOP bar).

`FileScanTaskId::data_file_path` remains `String` (owned identity key).

### Tests
- Existing split/merge/weight/serde tests green.
- New pin: `split_sub_tasks_arc_share_path_projection_deletes_and_predicate` —
  `Arc::ptr_eq` across N sub-tasks for path / projection / deletes / residual.
- New pin: `arc_fields_serialize_as_plain_string_and_arrays` — JSON shape + round-trip.

### Mid-unit gate (must pass before stop)

| Gate | Command | Exit |
|---|---|---|
| scan/task unit tests | `cargo test -p iceberg --lib scan::task` | **0** (24 passed, incl. new Arc + serde pins) |
| scan module (extra) | `cargo test -p iceberg --lib scan::` | **0** (194 passed) |
| clippy lib | `cargo clippy -p iceberg --lib -- -D warnings` | **0** |

### map.md
`crates/iceberg/src/scan/map.md` updated: task Arc-share + context residual Arc-clone (FK2.1).

### Critic-octo FK2.1 (8 cycles) — OCTO-CONVERGED
**Tip:** 65eab099  
Scratch: `/tmp/critic-octo-fk2_1-2026-08-08/`

| Cycle | Finding | Fix |
|------:|---|---|
| 1 | S2 offsets-aware Arc share unpinned; co-partition residual Arc unpinned | pins in `task.rs` + residual tests |
| 2 | S2 serde STOP bar lacked frozen golden | `arc_fields_json_matches_pre_arc_golden_bytes` |
| 3–4 | S3 plan-wide projection Arc residual; mutation-RED docs | ledger seeds |
| 5–7 | concurrency / reader / changelog alias attacks | no OPEN ≥ S1 |
| 8 | gate re-proof | OCTO-REPORT |

### Not in this unit (FK2.1)
- FK2.2 / FK2.3
- Cargo.toml (frozen)
- Delete-index / planning overlap

---

## FK2.2 — overlap delete/data manifest planning (scout #14)

**Worktree:** `/tmp/iceberg-rust-fk2_23`  
**Base tip:** `23867023` (includes FK1, FK2.1, FK3, FK4.1)

### Change
`crates/iceberg/src/scan/mod.rs` `TableScan::plan_files`:

- **Removed** the correctness-unnecessary barrier that `.await`ed the delete-entry
  processor before starting the data-entry processor.
- Delete-entry + data-entry streams now run **concurrently** (both fire-and-forget
  `spawn`s, same shape as the pre-existing data path).
- Data-entry processing that reaches `into_file_scan_task` parks on the existing
  `DeleteFileIndex` `Notify` until the index is published (all delete senders drop →
  populate task collects → `PopulateGuard::publish` → `notify_waiters`).
- Plan latency approaches `max(T_del, T_data)` instead of `T_del + T_data`.

### Hang-test design (inject-only, bounded timeouts; NO loom, NO stress harness)

| Pin | Hang class | Mechanism |
|---|---|---|
| `test_fk2_2_get_deletes_does_not_lose_wakeup_when_publish_races` | lost-wakeup | Concurrent waiter on `get_deletes_for_data_file`; yield to arm under read lock; publish via `PopulateGuard`; 5s timeout must complete |
| `test_fk2_2_failed_populate_wakes_concurrent_waiters_with_typed_error` | failed-populate | Two concurrent waiters; inject `Failed` via production publisher; both must error (not hang) under 5s |
| `test_fk2_2_sender_drop_publishes_and_wakes_waiter` | natural concurrent path | Send one delete context, concurrent waiter, drop sender → populate publishes → waiter gets the delete |
| Pre-existing SAF-007 suite | lost-wakeup + dead/never-polled/unwind populate | `test_waiter_is_armed_*`, `test_dead_populate_*`, `test_never_polled_*`, `test_unwinding_*` |

Mutation RED: arming `Notified` after releasing the read lock → lost-wakeup timeout; removing
`Failed` terminal → hang.

### Public API
None. Internal plan-path concurrency only.

---

## FK2.3 — delete-index keys (scout #15)

### Change
`crates/iceberg/src/delete_file_index.rs` `PopulatedDeleteFileIndex`:

- Partition maps keyed by **`(partition_spec_id, Struct)`** (`PartitionDeleteKey`) — was
  `Struct` alone with a post-filter linear `spec_id` compare.
- Global / eq-partition / pos-partition / pos-path lists **sorted by data sequence once**
  at build (`sort_deletes_by_sequence`).
- Lookup uses **`partition_point`** via `applicable_eq_deletes` (`delete_seq > data_seq`)
  and `applicable_pos_deletes` (`delete_seq >= data_seq`) — Java `findStartIndex` shape.
- Wrong key = delete resurrection across evolved specs that share partition values.

### Multi-spec identical-result-set pins

| Pin | Bar |
|---|---|
| `test_fk2_3_multi_spec_identical_result_sets_no_cross_spec_resurrection` | Same partition tuple under specs 1 and 2; each data file gets ONLY its own spec's eq+pos deletes; cross-spec paths must not appear |
| `test_fk2_3_multi_spec_seq_sorted_partition_point_identical_sets` | Multi-seq tails under two specs; insertion order reversed at build; applicable sets at `data_seq=4` match exact ordered paths (eq `>` / pos `>=`) |
| Pre-existing | `test_partition_scoped_pos_delete_still_requires_matching_partition_and_spec`, different-spec empty set in `test_delete_file_index_partitioned` |

Mutation RED: key by `Struct` alone → cross-spec resurrection; off-by-one on eq/pos seq
predicate → boundary delete moves; skip sort-at-build → partition_point wrong on reversed insert.

### Public API
None. `DeleteFileIndex` remains `pub(crate)`.

---

## Mid-unit gate (FK2.2 + FK2.3)

| Gate | Command | Exit |
|---|---|---|
| delete_file_index | `cargo test -p iceberg --lib delete_file_index` | **0** (28 passed, incl. FK2.2 hang + FK2.3 multi-spec pins) |
| scan module | `cargo test -p iceberg --lib scan::` | **0** (196 passed) |
| clippy lib | `cargo clippy -p iceberg --lib -- -D warnings` | **0** |

### map.md
`crates/iceberg/src/scan/map.md` updated: plan_files concurrent delete/data (FK2.2); delete-index
composite keys + hang/resurrection failure modes (FK2.3).

### Critic-octo FK2.2+2.3 (8 cycles) — see OCTO-REPORT
Scratch: `/tmp/critic-octo-fk2_23-2026-08-08/`

### Not in this unit
- FK4.2 / FK5
- Cargo.toml (frozen)
- Plan-wide projection Arc residual (FK2.1 S3 seed)
- Hour-0 plan_files wall-time bench (structural win; measure deferred)
