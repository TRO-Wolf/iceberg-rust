# FK1 — eq-delete columnar keyset (scout #3)

**Base:** `a966055e` (#182)
**Branch:** `feat/fk-mor-perf-campaign`
**Tag:** `[fork]`

## Hour-0 / after (debug test profile, same machine)

| matrix | BASE ns/row | AFTER ns/row | ratio |
|---|---:|---:|---:|
| 1M data × 100k del Long | 953.92 | 354.42 | 0.37× (~2.7× faster) |
| 1M data × 1M del Long | 1076.57 | 374.05 | 0.35× (~2.9× faster) |

Hot-path drop confirmed → ships (no minimum threshold).

## Change
- `EqDeleteKeySet` stores `KeyStore::I64(HashSet<i64>)` for single integer-like keys, else
  length-tagged `HashSet<Vec<u8>>` encodings — no probe-side `Vec<Option<Datum>>` clone storm.
- Probe encodes/hashes from Arrow arrays (typed downcasts).
- Parse path: when set-eligible, collect tuples first then build survival predicate once
  (null-batch fallback still needs the tree); when ineligible, predicate-only as before.
- Floats stay type-gated out; null data still bails to predicate (A answers).

## Soundness gates (mutation-RED at tip)
- Null bail: mutate `null_count() > 0` → `test_h6_set_returns_none_when_key_column_has_null` RED
- Float gate: admit Float/Double in `is_eligible_type` → `test_h6_gate_excludes_float_*` RED
- Full H6 harness: **24/24 green** (critic-octo expanded type matrix + null-only pins)

## Critic-octo FK1 (8 cycles) — soundness fixes
- **C1:** `delete_mask` null bail **before** empty short-circuit (I64 drops null deletes → empty store)
- **C2:** `i64_dropped_null_deletes` so `is_empty()` stays false for null-only I64; apply seam
  (`eq_delete_keep_mask`) always calls `delete_mask` (no empty-skip → keep-all)
- **C3–C7:** coverage pins (Uuid, Int, Timestamp/tz/ns, multi-col null-only Bytes) + i64 len guard
- **C4/C8:** mutation RED re-proven at tip for null bail + float gate

## Java cite (actor-found, 1.10.0 surface)
- Membership model: Iceberg `StructLikeSet` / equality-delete apply path (hashed set vs
  per-row predicate). Float exclusion rationale remains Arrow total-order vs Datum
  `OrderedFloat` signed-zero collapse (`test_h6_naive_set_diverges_on_negative_zero`).
- Follow-up seed: float fast-path with Java `Comparators` hashing + null-aware set membership.

## Residuals / seeds
- Null-aware columnar membership (drop null bail)
- Float/Double Java-Comparator hashing with bytecode class+method cite
- True deferral of survival-predicate materialization until first null-batch fallback
  (tonight still builds predicate at parse for fallback readiness)
- (S2 residual) Dictionary/Utf8View probe hard-error vs predicate fallback — pre-FK1 class
