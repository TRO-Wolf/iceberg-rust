# FK5 — `_pos` projection pushdown (scout #16)

**Branch:** `feat/fk-mor-perf-campaign`  
**Tag:** `[fork]`  
**Base (campaign):** `a966055e` (#182)  
**FK4.2 tip before this unit:** `b6856fad`  
**Worktree:** `/tmp/iceberg-rust-fk5`  
**CORRECTNESS-ADJACENT:** wrong `_pos` corrupts WRITTEN position deletes (RePark MERGE identity).

## Ship policy (user-locked)

1. **Streaming-only half first** — do not whole-file `try_collect` when projecting `_pos` (memory win alone justifies).
2. **Selection-aware ordinal half** — ONLY if mutation-RED-proven. If accounting gap cannot be guaranteed → **STOP** with exact gap named; streaming half still ships.
3. Prefer partial+STOP over wrong ordinals. **FK5 rushed is unacceptable.**
4. Bar: `_pos` oracle — dense + sparse pos-deletes + residual filters; row sets AND `_pos` values vs unpruned baseline; mutation-proven.
5. One cheap fork-side MERGE-shaped pin: write position deletes from synthetic mutation over a pruned/streamed scan, verify positions.
6. RePark MERGE battery is **OUT** of done-bar (post-repin).

## Design (locked for this unit)

| Half | Status | Notes |
|---|---|---|
| **Streaming (decode still sequential, no pushdown)** | **SHIP TARGET** | Still no `RowFilter` / `RowSelection` / RG prune when `_pos` projected; stream Parquet batches through transform + `survival_mask` instead of `try_collect` + eager `Vec`. |
| **Selection-aware ordinals (restore pushdown)** | **STOP unless RED-proven** | See [STOP — selection-aware ordinals](#stop--selection-aware-ordinals-exact-gap). |

### Streaming contract

When `_pos` is projected on the Parquet path:

1. Decode file **in physical order** with **no** row-skipping pushdown (same as pre-FK5).
2. Do **not** materialize the full file into a `Vec<RecordBatch>` before transform/delete apply.
3. Per batch: `RecordBatchTransformer` assigns `_pos` from a running physical counter (`next_row_position`), then `survival_mask` applies pos-deletes / residual / eq-deletes using the same absolute base, then filter.
4. Advance physical counters by the **full pre-filter** batch row count (transformer already does this; survival path must match).
5. Avro/ORC still go through `finish_whole_file_scan_task` (decode already materializes); no behavior change required for the memory win on Parquet MoR identity scans.

### Files

| Path | Change |
|---|---|
| `crates/iceberg/src/arrow/reader.rs` | replace `_pos` `try_collect` branch with streaming finish; share per-batch apply with whole-file tail |
| `task/fk5-pos-projection-ledger.md` | this ledger |

## STOP — selection-aware ordinals (exact gap)

**Not shipping selection-aware ordinals in this unit.** Exact accounting gap:

> **Parquet `RowFilter` does not expose physical file ordinals of survivors.** When a residual (or eq-delete predicate) is pushed as an `ArrowPredicate` `RowFilter`, the async reader returns only rows that evaluate true; skipped/non-matching rows are never delivered and **no per-row physical index is attached**. A running counter over *delivered* rows would assign dense 0..N-1 among survivors, which is **not** the Iceberg file ordinal required for position deletes.

Related gaps that must all close together (any incomplete subset is wrong `_pos`):

1. **`RowFilter` residual / eq-predicate pushdown** — no physical ordinal stream (primary gap above). Workaround would be: never push residual via `RowFilter` when projecting `_pos`; apply residual post-decode only (streaming half already does this). Restoring residual pushdown needs either (a) a reader API that yields (physical_pos, row) or (b) page/RG accounting that reconstructs ordinals without seeing dropped rows — neither exists in arrow-rs Parquet today in a form we can mutation-prove end-to-end.
2. **`RowSelection` pos-delete skips** — in principle trackable as Select/Skip spans with absolute base advances, *if and only if* residual is **not** also applied as `RowFilter` on the same builder (intersection of selection + filter still loses ordinals of filter-dropped rows). Multi-RG + byte-range split + page index make the Select/Skip walk easy to get subtly wrong.
3. **Row-group / page prune** — skipped RGs/pages require metadata row counts to advance the absolute base; wrong base shifts every subsequent `_pos`.
4. **Within-file split expand** — already suppressed when `_pos` is projected (`scan/mod.rs`); re-enabling would need per-split absolute base = sum of prior RG row counts, not 0.

**Decision:** STOP ordinal half. Ship streaming-only. Revisit only with a design that mutation-proves physical ordinals under at least residual + dense/sparse pos-deletes without relying on undelivered-row reconstruction from `RowFilter`.

## Pins (done-bar)

| Pin | Claim |
|---|---|
| multi-batch `_pos` continuity | small `batch_size` → `_pos` continues across batches (0..N-1 physical) |
| dense pos-delete + `_pos` | every-other-row deletes; survivors keep true physical `_pos` |
| sparse pos-delete + `_pos` | few deletes mid-file / multi-RG; survivors keep true physical `_pos` |
| residual + pos-delete + `_pos` | residual post-decode AND pos-deletes; row set **and** `_pos` match unpruned baseline oracle |
| mutation RED | break absolute-pos advance (or start base) → oracle RED |
| MERGE-shaped pin | scan `(_file,_pos)` under streaming path → write position deletes → MoR omits exactly those rows |

## Critic-octo

Scratch: `/tmp/critic-octo-fk5-2026-08-08/`  
8 cycles, `early_stop=false`. Soundness gates must mutation-RED within octo.

## Residuals

- Selection-aware ordinals (STOP — gap named above)
- Avro/ORC true streaming decode (out of scope; decode already whole-file)
- RePark MERGE battery post-repin (hub)

## Hour-0 / after

Streaming is a memory / peak-RSS win (no full-file `Vec` of decoded batches on the `_pos` path). No CPU threshold required. Record qualitative claim: peak RSS / decoded-batch residency drops from O(file) to O(batch) for Parquet `_pos` scans.

## Outcome

**SHIPPED:** streaming-only half  
**STOP:** selection-aware ordinals (exact gap named above — `RowFilter` does not expose physical ordinals of undelivered rows; Select/Skip + multi-RG + residual intersection remains incomplete without that)

### Code
- `process_parquet_file_scan_task`: `_pos` branch → `stream_pos_projection_scan_task` (no `try_collect`)
- Shared: `build_scan_task_transformer`, `resolve_whole_file_delete_context`, `apply_pos_aware_batch`
- Avro/ORC `finish_whole_file_scan_task` uses the same per-batch apply

### Pins green
- `fk5_pos_projection_multi_batch_continuity`
- `fk5_pos_oracle_dense_pos_deletes`
- `fk5_pos_oracle_sparse_pos_deletes_multi_rg`
- `fk5_pos_oracle_residual_and_pos_deletes`
- `fk5_pos_mutation_absolute_pos_advances_by_full_batch`
- `fk5_merge_shaped_pos_delete_from_streamed_identity_scan`
- pre-existing `test_scan_projects_pos_metadata_column` + avro/orc pos-delete suite

### Mutation RED (pre-commit)
Advance `absolute_pos` by **filtered survivor count** instead of pre-filter `row_count` in `apply_pos_aware_batch` →  
`fk5_pos_mutation_absolute_pos_advances_by_full_batch` **FAILED**:
```
left:  [(2, 2), (4, 4), (5, 5)]
right: [(2, 2), (3, 3), (4, 4), (5, 5)]
```
(batch-1 pos-delete base shifted; over-deleted id 3). Restored before commit.

Note: `_pos` *column* values are assigned by `RecordBatchTransformer::next_row_position` (full-batch advance, independently tested); `absolute_pos` drives pos-delete `survival_mask` base. Both must stay physical-ordinal-aligned.

### Critic-octo
See `/tmp/critic-octo-fk5-2026-08-08/OCTO-REPORT.md`
