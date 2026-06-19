<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Audit Review Plan — repo vs. roadmap (2026-06-19)

> **What this is.** A standing list of *candidate* work items surfaced by a deep read-only audit of the
> whole repository against the long-term parity goal. These are **proposals to triage**, not committed
> tasks — promote the ones worth doing into [task/todo.md](todo.md) when a block is planned. Nothing here
> is a code change yet; no files were edited to produce this plan.

## Status log

- **2026-06-19** — **P1 (SEC-01/02/03) DONE**, merged as #98 (`fix/untrusted-input-panic-hardening`):
  Puffin footer reader, temporal `Datum` Display, and `SqlCatalog` pool-config all hardened against
  panic-on-bad-input; the temporal fix also repaired a latent pre-1970 negative-remainder panic. Gate
  green (iceberg lib 2591/0, sql lib 72/0). **QUAL-01** (`tracing` dep + wiring the 3 silent-swallow
  sites) is **next**, with the dependency-add approved 2026-06-19. Everything else below remains open.

## Provenance & method

- **Date:** 2026-06-19. **Base:** `main` @ `7a6ef42c` (events/listeners #97 merged).
- **Method:** three independent Opus auditors (max-reasoning, read-only), one per lens —
  **parity-truth**, **security/correctness**, **quality/maintainability** — then orchestrator synthesis.
  The single HIGH finding (SEC-01) was independently re-verified against the source by the orchestrator
  before landing here.
- **Calibration note (read this first):** the headline verdict across all three lenses is that the repo
  is in **genuinely good shape** — honest matrix, real interop harness, defensively-written new code.
  This list is deliberately the *exception report*. It skews toward small, real, well-evidenced items;
  it is **not** evidence of systemic problems. Strengths are recorded at the bottom so the plan is balanced.

## Verdict by lens

| Lens | Verdict | Findings |
|---|---|---|
| **Parity-truth** | Capability-truth is high; every ❌ is genuinely absent and sampled ✅ rows are real (code + tests, not stubs). Issues are **documentation-truth**, not capability-truth. | 5 (2 med / 2 low / 1 info) |
| **Security/correctness** | Core data paths are defensively written; the one real gap is the **Puffin metadata reader** (no size validation), plus a few narrower panic-on-bad-input spots. No active secret leak. | 6 (1 high / 2 med / 3 low) |
| **Quality/maintainability** | Strong test depth and honest docs; headline debt is **no `tracing` in the core crate** (silent swallows), one **verbatim copy-paste** (ORC↔Avro scan task), and an **obsolete `dead_code` blanket**. | 7 (3 med / 3 low / 1 info) |

## Priority summary

| # | ID | Title | Sev | Effort | Approval gate |
|---|---|---|---|---|---|
| **P1 — correctness at the untrusted-input boundary** ||||||
| 1 | SEC-01 | Puffin metadata reader underflows on truncated/hostile footer | high | M | — |
| 2 | SEC-02 | Time/Timestamp `Datum` Display panics on out-of-range stored value | med | S | — |
| 3 | SEC-03 | `SqlCatalog` construction panics on non-numeric pool config | med | S | — |
| **P2 — maintainability & observability** ||||||
| 4 | QUAL-01 | Core crate has no `tracing`; commit/cleanup failures are silent | med | S | **dep add** |
| 5 | QUAL-02 | `process_orc_file_scan_task` is a ~125-line verbatim clone | med | M | — |
| 6 | QUAL-03 | Obsolete module-level `#![allow(dead_code)]` on ORC/Avro readers | low | S | — |
| **P3 — doc-truth resync (batchable into one doc PR)** ||||||
| 7 | PAR-01 | Roadmap status prose is one resync behind the matrix | med | S | — |
| 8 | PAR-03 | GAP_MATRIX cells regrew into multi-paragraph narratives | med | M | — |
| 9 | PAR-02 | Test-count / audit-date provenance stamps stale ~15 PRs | low | S | — |
| 10 | PAR-04 | Documented base `arrow 57.1` drifted to `57.3` | low | S | — |
| **P4 — low / info (opportunistic or track-only)** ||||||
| 11 | SEC-04 | `S3Config` derives `Debug` over plaintext secrets (latent leak) | low | S | — |
| 12 | SEC-05 | Scan path `.unwrap()` on oneshot recv & poisoned Mutex | low | S | — |
| 13 | SEC-06 | `RollingFileWriter` status accessors panic before first write | low | S | — |
| 14 | QUAL-04 | 5 public TypeUtil fns with no in-tree caller (doc-note only) | low | S | — |
| 15 | QUAL-05 | Hand-rolled ORC footer parser = future dep-sync trap (track) | low | S | — |
| 16 | QUAL-06 | `notify_all` carries a `cfg(test)` dispatch gate (fragile seam) | low | M | — |
| 17 | QUAL-07 | ~27 stale local branches; `.claude/` not gitignored | info | S | — |
| 18 | PAR-05 | Confirm breadth-first vs. frontier (encryption/LockManager) order | info | S | **decision** |

---

## P1 — correctness at the untrusted-input boundary

### 1. SEC-01 (high) — Puffin metadata reader underflows on a truncated/hostile footer
- **Where:** `crates/iceberg/src/puffin/metadata.rs:201,218,300,333`; reached via `puffin/reader.rs:42`.
- **Problem (orchestrator-verified):** `read_footer_payload_length` does `input_file_length - FOOTER_STRUCT_LENGTH(12)` with **no check that the file is ≥ 12 bytes** (line 201); `read_footer_bytes` does `input_file_length - footer_length` where `footer_length` derives from a file-controlled `u32` (line 218); `read` slices `footer_bytes[len - magic_length..]` with no min-length guard (line 300). Siblings (`decode_flags`, `extract_footer_payload_as_str`) *do* use `.get(...).ok_or_else` — these three do not. No `overflow-checks` override, so this **panics in debug / wraps to a multi-exabyte read in release**.
- **Reachable from untrusted bytes:** PuffinReader is wired into the V3 deletion-vector read path (`arrow/caching_delete_file_loader.rs`) and table-statistics path. A corrupt/short Puffin file crashes a scan — a DoS/robustness hole, and behavior differs debug vs release.
- **Fix:** replace every length subtraction with `checked_sub(...).ok_or_else(|| DataInvalid)`; validate `input_file_length >= FOOTER_STRUCT_LENGTH` and `footer_length <= input_file_length` before use. Mirror the proven pattern in `delete_vector.rs::deserialize_deletion_vector_v1`. Add regression tests: a sub-12-byte file and a `footer_payload_length = u32::MAX` file.
- **Note:** the matching interop/sabotage discipline already exists in the DV decoder — this just brings the one lagging on-disk parser up to that bar.

### 2. SEC-02 (med) — Time/Timestamp `Datum` Display panics on an out-of-range stored value
- **Where:** `crates/iceberg/src/spec/values/temporal.rs:66,97,103`; `datum.rs:304`; `literal.rs:645`.
- **Problem:** `microseconds_to_time` does `NaiveTime::from_num_seconds_from_midnight_opt(...).unwrap()`; for a Time value < 0 or ≥ 86_400_000_000 µs the `opt` returns `None` → panic. `Datum::fmt`/`Literal` format pass the raw stored `i64` straight in with no clamp. Time/timestamp Datums originate from on-disk bytes (min/max stats, partition values, manifest entries), so a single corrupt stat panics any `to_string()`/tracing field/`inspect` query.
- **Fix:** make the conversions fallible and render a placeholder (`<invalid time: {val}>`) or surface `DataInvalid` instead of `.unwrap()`. Java does not panic when formatting an out-of-range value. Regression test: display a Time Datum with a negative and an `86_400_000_001` value.

### 3. SEC-03 (med) — `SqlCatalog` construction panics on non-numeric pool config
- **Where:** `crates/catalog/sql/src/catalog.rs:262,267,272`.
- **Problem:** `config.props.get("pool.max-connections").map(|v| v.parse().unwrap())` (and `pool.idle-timeout`, `pool.test-before-acquire`) — a malformed operator-supplied value (`"ten"`) aborts catalog construction with a bare-unwrap panic. Violates the "no bare `.unwrap()` in production paths / typed-error" contract.
- **Fix:** `.map(|v| v.parse()).transpose().map_err(|e| Error::new(DataInvalid, "invalid pool.<x>").with_source(e))?`. Unit test: a non-numeric `pool.max-connections` yields `Err`, not a panic.

---

## P2 — maintainability & observability

### 4. QUAL-01 (med) — Core `iceberg` crate has no `tracing`; failure paths are silent  ⟶ **dependency-approval gate**
- **Where:** `crates/iceberg/Cargo.toml` (no `tracing`/`log` dep); deferred sites at `transaction/mod.rs:461-483`, `events/mod.rs:52-55`, `metrics/mod.rs:23-24`, `transaction/expire_cleanup.rs`.
- **Problem:** CLAUDE.md mandates structured `tracing` in library code, but the core crate has **zero** `tracing` dependency and three comment-deferred log sites. `transaction/mod.rs:481` swallows a panicking commit listener via `let _ = catch_unwind(...)` with no log line; the cleanup-failure collector is likewise silent. Six other workspace crates already depend on `tracing`.
- **Fix:** add `tracing` to the iceberg crate and wire `tracing::warn!(?error, ...)` at the commit-listener swallow, the `LoggingMetricsReporter`, and the cleanup collector — one change closes all three deferrals.
- **Gate:** adding a dependency hits CLAUDE.md's "never edit dependency files without explicit approval." **Flag for approval; do not self-approve.**

### 5. QUAL-02 (med) — `process_orc_file_scan_task` is a ~125-line verbatim clone of the Avro path
- **Where:** `crates/iceberg/src/arrow/reader.rs:690-801` (avro) vs. `:815-925` (orc).
- **Problem:** the ORC scan task is, by its own doc comment, "a VERBATIM structural clone" of the Avro one — identical except the materialization call and three "Avro"/"ORC" strings. The delete-loading, schema build, `RecordBatchTransformer`, survival-predicate AND-ing, and positional-delete loop are byte-identical. A future fix to delete-application or schema-evolution must be made twice or the formats **silently diverge — a row-resurrection (data-correctness) class bug**, not cosmetic.
- **Fix:** extract the format-agnostic tail (everything after `let batches = <reader>(...).await?;`) into one `finish_whole_file_scan_task(...)` both paths call; drop the `avro_` prefix on the now-shared `build_*`/`survival_mask` helpers.

### 6. QUAL-03 (low) — Obsolete module-level `#![allow(dead_code)]` on the ORC/Avro readers
- **Where:** `crates/iceberg/src/arrow/orc_reader.rs:82-86`, `avro_reader.rs:90-95` (both wired now at `reader.rs:710/833`).
- **Problem:** the blanket is justified in-comment as a "pre-U2 stance," but U2 (scan wiring) is done — both readers are load-bearing. A module-wide `dead_code` allow permanently blinds the compiler to genuinely-dead helpers added later, exactly when these files most need the signal.
- **Fix:** remove the module-level allow; if `read_*_data_bytes` (the offline sync-test entry) is the only unused item, scope a targeted `#[allow(dead_code)]` to that fn or `#[cfg(test)]` it.

---

## P3 — doc-truth resync (batchable into one doc-only PR)

> These four are all documentation; none touches code. They could land as a single "resync docs to reality"
> PR. Per precedence (matrix wins on status, Roadmap is corrected), the **Roadmap** is the stale party.

### 7. PAR-01 (med) — Roadmap status prose is one resync behind the matrix
- **Where:** `Roadmap.md:162-164,250,254,259,342` vs. `GAP_MATRIX.md:116,117,122,137,142`.
- **Problem:** Roadmap "Missing ❌" still lists `BatchScan` and ORC/Avro DATA as missing and buckets `RewriteTablePath` into residue; the live matrix has `BatchScan` ✅ (#88), ORC/Avro read 🟡 (#91/#95/#96), `RewriteTablePath` 🟡 (#89), and events ✅ (#97) — none reflected. Roadmap current-state self-dates "resynced 2026-06-17," before #87–#97. A new session reading the Roadmap (mandated before the matrix) is told built+interop-proven capabilities are still missing.
- **Fix:** resync the "Missing ❌" line and Phase-3/Phase-4/headline-gap sections to the matrix; add events ✅; re-date the current-state block.

### 8. PAR-03 (med) — GAP_MATRIX cells regrew into multi-paragraph narratives
- **Where:** `GAP_MATRIX.md:74-79` (the cell-style rule) vs. rows 124/125/136/149/151.
- **Problem:** the matrix's own rule wants terse cells (location + 1–2 sentences + flip date + links), with narratives in `docs/parity/archive/`. Live: note column median ~1,618 chars, 41/68 rows > 1,000 chars, worst (row 136) ~19.8K. The whole matrix is ~80K tokens — exceeds one Read call, re-creating the doc-mass problem the 2026-06-09 hardening fixed. This is the "narratives regrow" risk the Roadmap itself names (line 360).
- **Fix:** compaction pass — move each cell's increment narrative into the existing `docs/parity/archive/2026-06_matrix-cell-narratives.md` and shrink the cell to location + 1–2 sentence status + flip date + archive link. Start with rows 136/125/151/124/149. Re-run the pipe-count audit (exactly 5 `|` per row) after.

### 9. PAR-02 (low) — Provenance: test counts & audit date stale ~15 PRs
- **Where:** `GAP_MATRIX.md:45-56,53`; `Roadmap.md:131-139`.
- **Problem:** both claim "measured 2026-06-13: 2,238 src tests" and "re-audited 2026-06-07/13," while ~15 flips inside the same matrix are dated 06-17…06-19 (the events row even cites "2584 green"). Live census ≈ 2,585 src / 2,713 incl. tests / 3,116 workspace. Source-of-truth list omits the new `events/` dir. No capability claim is wrong — only the meta-provenance.
- **Fix:** refresh the count line + audit date past 06-19, add `crates/iceberg/src/events` to the source list — or replace the hard-coded counts with the one-line census command so the number is reproducible. (Folds naturally into #8.)

### 10. PAR-04 (low) — Documented base `arrow 57.1` drifted to `57.3`
- **Where:** CLAUDE.md "Project snapshot"; `Roadmap.md:137`; `GAP_MATRIX.md:45-46`; `Cargo.toml:106`.
- **Problem:** three docs pin "arrow 57.1 / parquet 57.1"; `orc-rust 0.7` (added with approval in #95) resolves the tree to 57.3. The bump is intentional and benign (single arrow major, no v58 leak) — but the contract docs misstate the resolved base.
- **Fix:** update the three snapshots to "arrow 57.3 / parquet 57.3" with a one-line note (orc-rust 0.7 unified the patch level). No code change. (This was **not** a prohibition violation — #95 was user-approved.)

---

## P4 — low / info (opportunistic, or track-only)

- **11. SEC-04 (low) — `S3Config` derives `Debug` over plaintext secrets.** `io/storage/config/s3.rs:73,83,104` — `secret_access_key`/`session_token`/SSE customer key are printed by the derived `Debug`. No active leak today (grep found no `?config` log), but a future `debug!(?config)` would leak. **Fix:** hand-written redacting `Debug` (`"***"` for secret fields); audit gcs/azdls/oss siblings; test that the secret never appears in `format!("{config:?}")`.
- **12. SEC-05 (low) — Scan path bare `.unwrap()` on internal failure.** `arrow/reader.rs:526` (`delete_filter_rx.await.unwrap()` — panics if the delete-filter task is dropped/cancelled) and `:623` (`.lock().unwrap()` — panics on poison; the events module already uses `into_inner()` recovery). **Fix:** map `RecvError` to a typed error; recover the poisoned guard via `unwrap_or_else(|p| p.into_inner())`.
- **13. SEC-06 (low) — `RollingFileWriter` status accessors panic before first write.** `rolling_writer.rs:246,250,254` — `self.inner.as_ref().unwrap()` on a public `CurrentFileStatus` method while `inner` is `None` until the first record. **Fix:** return sentinel/Option, or a contextual `.expect(...)` naming the misuse.
- **14. QUAL-04 (low) — 5 public TypeUtil fns with no in-tree caller.** `spec/schema/{id_reassigner,utils,compat}.rs` — `assign_fresh_ids_with_base`/`reassign_doc`/`index_quoted_name_by_id`/`join`/`estimate_size` are pure, unit-tested, Java-faithful, but unexercised by any production path. **Do not delete** (parity surface). **Fix:** a one-line module doc note "parity-complete, no in-tree caller yet"; wire + integration-test when the create-table/metadata-join consumers land.
- **15. QUAL-05 (low, track) — Hand-rolled ORC footer protobuf parser is a future dep-sync trap.** `arrow/orc_reader/footer.rs` exists only because `orc-rust` drops the `iceberg.id` type attribute. It's well-built (bounds-checked, varint-guarded, `try_from` not `as`) but is a structural divergence we own forever, with explicit v1 scope limits (ZLIB-only footer, top-level primitives). **Fix:** keep the "why we parse it ourselves" rationale current; track to retire it if `orc-rust` exposes attributes upstream; confirm the v1 scope limits are stated in the matrix ORC row.
- **16. QUAL-06 (low) — `notify_all` carries a `cfg(test)` dispatch gate.** `events/mod.rs:130-162` short-circuits dispatch under `cfg(test)` via a thread-local ARM flag, relying on the `#[tokio::test]` current-thread scheduler keeping the emit on the arming thread. Production behavior is correct (compiles out), but a switch to a multi-thread test runtime or an emit site moving to a spawned task could silently turn event tests into no-ops. **Fix:** document/assert the current-thread assumption above the gate, or move arming to a scoped registry; add a regression test that catches an emit site leaving the arming thread.
- **17. QUAL-07 (info) — Repo hygiene.** ~27 stale local branches (squash-merged, so `--merged` hides them — the trap documented in `lessons.md`); `.claude/` and `task/sepmo-convergence.md` untracked. **Fix:** prune branches via the content-diff safety gate (`git diff <tip> <squash-commit>` empty ⇒ safe), **never** `--is-ancestor`; add `.claude/` to `.gitignore`; decide track-or-ignore for `task/sepmo-convergence.md`.
- **18. PAR-05 (info, decision) — Confirm sequencing intent.** The largest true remaining Java-core gaps are encryption (row 128, stub only), LockManager (127, zero code), geometry/geography V3 (87), and ORC/Avro **write** (116/117). Recent PRs delivered ORC/Avro *read* + events — real work, but lower on the project's own effort×value ranking. **Not a defect** — likely a deliberate breadth-first call. **Decision needed:** record in the Roadmap headline-gaps whether breadth-first is intended, so it reads as a decision, not drift; otherwise schedule the frontier work explicitly.

---

## Notable strengths (the balancing column)

The audit is an exception report; these are the things that are genuinely well-done and should be preserved:

- **Capability-truth is high.** Every ❌ row spot-checked has zero backing code; every recently-flipped ✅ (events, BatchScan, ORC/Avro read, ConvertEqualityDeleteFiles) has real wired code + passing tests, not stubs. The "two ✅ bars" disclaimer is honest — 25/36 greens cite an interop oracle, the rest are labeled inherited-and-usable.
- **The interop harness is real and broad** — 38 driver scripts, all 9 named Java oracle classes present, 40+ `interop_*.rs` tests, env-gated to skip cleanly offline and run real bidirectional Java round-trips with fail-closed sabotage batteries.
- **The untrusted-input boundary is mostly exemplary.** The DV V1 decoder validates length-vs-buffer, rejects hostile counts before allocating, checks CRC, and carries `u32::MAX` regression tests; the DV encoder enforces the 2GB bound before its single allocation; the ORC footer parser is bounds-checked, varint-overflow-guarded, and uses `try_from` over `as`; the Avro/serde decimal/fixed path validates width before copying. (SEC-01 Puffin is the lone exception — which is why it's P1.)
- **The events registry is correctly designed** — clones listeners under the read lock and drops the guard before invoking callbacks (reentrancy-safe, proven by test), recovers from lock poison via `into_inner()`.
- **No active secret-logging path was found** — glue/s3tables pass credentials straight into the AWS SDK without Debug-logging; SEC-04 is latent (a missing guard rail), not an active leak.
- **Test depth ships with the code** — big files are dominated by inline tests (row_delta.rs ~1,200 code / ~5,900 test); the matrix cells are unusually honest about residue (named uncalled fns, interop-N/A exemptions, silent-swallow deferrals); the `transaction/map.md` is current with the code it indexes.
- **The ActionsProvider factory** mirrors Java's throw-by-default 12-method shape with a typed `FeatureUnsupported` default — no faked or dead provider methods.

## Suggested execution order

1. **P1 as one small "harden untrusted-input boundary" PR** (SEC-01 + SEC-02 + SEC-03) — all are bounded, test-backed, no dep changes; SEC-01 is the only HIGH and is confirmed real.
2. **P3 as one doc-only "resync to reality" PR** (PAR-01/02/03/04) — zero code, restores the matrix/Roadmap as trustworthy oracles; do the matrix compaction (PAR-03) carefully with a post-edit pipe-count audit.
3. **P2 next** — QUAL-02 (dedup, prevents a data-correctness drift) and QUAL-03 (S, trivial); QUAL-01 only after the **`tracing` dependency is explicitly approved**.
4. **P4 opportunistically** — fold SEC-04/05/06 into a future writer/IO touch; QUAL-04/05 are doc-notes; QUAL-07 is repo hygiene; **PAR-05 needs a one-line decision from the plan-owner** before it becomes anything.
