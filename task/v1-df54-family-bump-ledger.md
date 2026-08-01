# V1 — DF 54 family bump ledger

**Unit:** V1 FORK DF-54 FAMILY BUMP  
**Branch:** `chore/df54-family-bump`  
**Base:** G1 tip `bc4ffa19` (stacked; merge-order: G1 first)  
**Floors:** datafusion 54.1.0 / arrow*+parquet 58.4 / package identity 0.9.1  

## Coupled deps (family-adjacent)

| Dep | Change | Why |
|---|---|---|
| `home` | `=0.5.11` → `0.5.12` | datafusion-cli 54 / rustyline |
| `orc-rust` | 0.7 → 0.8 | arrow 58 only |
| `serde_arrow` feature | arrow-57 → arrow-58 | dual-version RecordBatch break |
| `sqllogictest` | 0.28.3 → 0.29 | match datafusion-sqllogictest 54 |
| `rust-toolchain` | nightly-2025-10-27 → nightly-2026-03-05 | AWS SDK / fastnum MSRV 1.94+ |
| `rust-version` | 1.92 → 1.94 | align MSRV |

## Cherry-pick / hand-port

| Upstream SHA | Applied how |
|---|---|
| `477a1e525` (#2206 DF53+Arrow58) | Pattern port (floors, row-group API, date deprecations) |
| `875fdb746` (#2648 DF54) | Pattern port (`as_any` strip, Cast.field, Arc PlanProperties) |
| `e8460eee` (#2872 DF54.1) | Floors 54.1.0 |
| G1 / delete / write / ScanKnobs | Hand-port compile fixes only |

## Status

- Actor build: in progress → cargo check --workspace --all-targets green (pre-test)
- Octo: pending

## Actor gate (pre-octo)

| Gate | Result |
|---|---|
| `cargo check --workspace --all-targets` | green |
| `cargo test -p iceberg --lib --all-features` | **3068 passed** |
| `cargo test -p iceberg-datafusion --all-features` | green (incl. nested insert re-pin + doctest) |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | green (new nightly lints auto-fixed) |
| `make interop` | deferred to octo / pre-push (needs JDK/mvn) |

### Behavior re-pins (cited)

1. **test_insert_into_nested** — DF54 field-aware `CastExpr` + `validate_field_compatibility` rejects nullable SQL `named_struct` leaves → non-null nested Iceberg required fields. Fixture leaves under `address` re-pinned OPTIONAL; zip literal cast to INT; expect! snapshot updated. Citation: DF 54 upgrade guide (CastColumnExpr → field-aware CastExpr); datafusion-common `nested_struct.rs` nullability rule. Required-nested SQL insert follow-up out of family-bump scope.

## Octo

Pending: 8× critic-octo early_stop=false

## Final gate (OCTO-CONVERGED)

| Gate | Result |
|---|---|
| make check | **green** |
| cargo test -p iceberg --lib --all-features | **3068 passed** |
| cargo test -p iceberg-datafusion --all-features | **green** |
| make interop | **52 passed, 0 failed** (floor 52) |
| cargo audit | quick-xml RUSTSEC-2026-0194/0195 **survive** (versions 0.37–0.39; fix ≥0.41 unreachable) |
| critic-octo 8× early_stop=false | **OCTO-CONVERGED** |

### Residual product seeds (not ship blockers)
1. **SEED-df54-required-nested-sql-insert** — required nested leaves + SQL named_struct need engine cast path under DF54 field-aware CastExpr.
2. **quick-xml ≥0.41** — when object_store unlocks, remove advisories.

### Tip SHA
`cbf1d2f5fe6e1ff053daffd8402716a75be8d91a` (branch HEAD at OCTO-CONVERGED push; if advanced by docs-only tip-stamp, use `git rev-parse origin/chore/df54-family-bump`)
