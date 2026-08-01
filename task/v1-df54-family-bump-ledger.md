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
