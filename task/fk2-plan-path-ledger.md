# FK2 — plan-path ownership + overlap (scout #5 + #14 + #15)

**Branch:** `feat/fk-mor-perf-campaign`  
**Tag:** `[fork]`  
**Base (campaign):** `a966055e` (#182)  
**FK1 tip before this unit:** `c7b14d80` (OCTO-CONVERGED)

## Sequencing
FK2 is three cuts; **strict serial** per campaign brief:
1. **FK2.1** Arc-share `FileScanTask` innards (scout #5) — this unit
2. FK2.2 overlap delete/data manifest planning (scout #14) — **after FK3 / FK4.1**
3. FK2.3 delete-index keys (scout #15) — **after FK3 / FK4.1**

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
**Tip:** `ad196f4f`  
Scratch: `/tmp/critic-octo-fk2_1-2026-08-08/`

| Cycle | Finding | Fix |
|------:|---|---|
| 1 | S2 offsets-aware Arc share unpinned; co-partition residual Arc unpinned | pins in `task.rs` + residual tests |
| 2 | S2 serde STOP bar lacked frozen golden | `arc_fields_json_matches_pre_arc_golden_bytes` |
| 3–4 | S3 plan-wide projection Arc residual; mutation-RED docs | ledger seeds |
| 5–7 | concurrency / reader / changelog alias attacks | no OPEN ≥ S1 |
| 8 | gate re-proof | OCTO-REPORT |

### Not in this unit
- FK2.2 / FK2.3
- Cargo.toml (frozen)
- Delete-index / planning overlap
