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

# Unit brief — G3: adjacent-split merge, interop pin (row R148)

**Prerequisite:** `fix/scan-plan-adjacent-split-merge` (`b38d71a2`) MERGED to fork `main`.
Branch this unit off the merged tip. Execution mode: **AC·OO, both Opus at high/max effort,
independent Critic** (per CLAUDE.md `<subagent_policy>`). Mode A — one scoped PR.

`dev/java-interop/README.md` is the authoritative harness contract (directions, fixture flow,
comparison semantics). This brief scopes WHAT to prove, not HOW the harness works.

## Why this unit exists

The merge port landed with **offline unit tests only** (7 tests in `scan/task.rs` +
`scan/task_group.rs`, mutation-proven RED). The interop layer — the only 1:1 evidence this repo
accepts for a ✅ row — does **not** exercise it.

**Precise statement of the oracle gap.** The scan-plan oracle is *not* count-only: `member_key`
is `(basename, start, length)` and each group is compared as a sorted member-key set, normalized
into an order-insensitive, duplicate-preserving multiset. A merge divergence *is* representable
in the comparison — Rust's `[(big,0,512),(big,512,512)]` vs Java's `[(big,0,1024)]` are unequal
sets. The gap is **fixture coverage**: with the current fixture and knobs (target 4096 / lookback
5 / open-file-cost 0), no bin ever deterministically holds ≥2 *adjacent* splits of one file, so
the merge branch is never taken on either side and the comparison is silently vacuous with
respect to it. The nightly runner reached that configuration by accident — delete-file path
length shifted pack weights across the 4096 knife edge — which is exactly why the failure was
runner-only and not locally reproducible.

**The lesson class to record:** a bidirectional oracle can compare the right *observable* at the
right *granularity* and still prove nothing about a code path its *fixture* never reaches. Assertion
granularity and fixture coverage are independent axes; only the first was audited when R148 first
went ✅.

## Proposition ledger

- **M-1 (Direction 1, Java→Rust).** Over a Java-written table containing the new merge fixture,
  Rust `plan_tasks` emits a group whose member for that file is a **single spanning
  `(basename, 0, fileLength)`** — identical to Java's `planTasks()` member, where Java's merge is
  performed by the real `BaseCombinedScanTask(List)` → `TableScanUtil.mergeTasks`.
- **M-2 (Direction 2, Rust→Java).** Over the Rust-written table, Java's real `planTasks()` produces
  the same single spanning member as Rust's own `rust_scan_plan.json`.
- **M-3 (Non-vacuity — the load-bearing guard).** *(Superseded in execution: the shipped
  discriminator is the exact plan-SHAPE assert read against the offsets-aware-split invariant; the
  span inequality below is a degenerate-fixture guard — see task/lessons.md 2026-07-24.)* The fixture is proven to have *actually exercised
  the merge branch*: at least one emitted member's `length` **strictly exceeds** the largest single
  row-group span of its source file (per the manifest's field-132 split offsets). A member equal to
  a single row-group span means no merge occurred and the pin is vacuous.
- **M-4 (Adjacency is respected, not just coalescing).** A same-file pair that is co-binned but
  **non-contiguous** does NOT merge. Java's `mergeTasks` is a single-pass adjacent-run collapse, not
  a group-by-file. Prove this survives the round trip rather than only in the offline unit test.
- **M-5 (Fail-closed sabotage).** A third sabotage leg that makes the merge non-load-bearing must
  make the plan DIVERGE. HARD-FAIL, never SKIP (CLAUDE.md working conventions); restore any `.bak`
  first, capture the mutator's exit with `|| rc=$?` under `set -euo pipefail`.

## Fixture design — the determinism requirement

The existing `big.parquet` is the wrong vehicle: its row-group *offsets* vary with the parquet-mr /
parquet-rs build, and it carries `big-deletes`, whose bytes enter the bin-pack weight — that pair is
precisely the 4096 knife edge that made the nightly non-reproducible. Do not tune it.

Add a **dedicated fixture file** (suggested `merge.parquet`) with two properties:

1. **≥2 row groups** (tiny parquet row-group size ⇒ split offsets ⇒ offsets-aware split), and
2. **total file length comfortably below the target**, with **no delete file attached**.

Property (2) is what buys determinism: if the file's whole length fits in one bin, *all* of its
row-group splits necessarily co-bin regardless of how the offsets fall, so the merge fires on every
parquet build. Note the pleasing consequence — **the merged member key is
`(merge.parquet, 0, fileLength)`, which is entirely independent of the internal row-group grid.**
Merging *removes* the environment sensitivity that made the original failure runner-only, so this pin
is strictly more robust than the fixture it supplements.

Guard the sizing rather than assuming it: assert the file's length + any weight contribution is
`< TARGET`, so a future fixture edit that pushes it over fails loudly instead of silently reverting
M-1 to vacuity.

Both sides must build the fixture identically, with the knobs hand-declared on each side
(anti-circular): `InteropOracle.ScanPlanOracle.{TARGET,LOOKBACK,OPEN_FILE_COST}` mirror
`interop_scan_plan.rs` `{TARGET,LOOKBACK,OPEN_FILE_COST}`.

## Files in scope

- `crates/iceberg/tests/interop_scan_plan.rs` — fixture write, M-1/M-3/M-4 assertions.
- `dev/java-interop/src/main/java/org/apache/iceberg/InteropOracle.java` — `ScanPlanOracle`
  generate / verify / sabotage; M-2 and the new sabotage leg.
- `dev/java-interop/run-interop-scan-plan.sh` — chain step count (6 → 7 if a leg is added), header
  fixture doc.
- `docs/parity/GAP_MATRIX.md` — R148 cell: the 2026-07-24 merge paragraph currently rests on unit
  tests; append the interop attestation. Run `make check-matrix-anchors`.
- `task/lessons.md` — the coverage-vs-granularity lesson above.
- `dev/java-interop/map.md` — lockstep if the chain shape changes.

**Stale-citation cleanup (in scope, small):** `run-interop-scan-plan.sh` still cites **`row 146`** at
lines 20 and 134. R146 is conflict-detection; this harness is **R148**. The merge branch fixed the
`.rs` and `map.md` but not the `.sh`.

## Out of scope

- The `planTaskGroups(groupingKeyType)` partition-aware overload (still deferred on R148).
- `DataTask` / `ContentScanTask<F>` exposure (R148's standing deferrals).
- Re-litigating the path-vs-identity file proxy (below) — flag only.

## Known residue to carry forward, not fix here

The shipped merge uses **data-file path** as the file-identity proxy where Java uses **reference
equality on the `DataFile` instance** (`BaseFile`/`GenericDataFile` override neither `equals` nor
`hashCode` — javap-confirmed). The merge-branch Critic disproved divergence as unreachable under the
one-manifest-entry-per-live-path invariant. That is sound for well-formed plans but is an *invariant*,
not a type-level guarantee — Java forecloses the same class structurally by leaving whole-file
`BaseFileScanTask` off `MergeableScanTask`. Record it as named residue on R148; do not widen this
unit to add split provenance.

## Done gate

`typos . && cargo fmt --all -- --check && cargo clippy --all-targets --all-features --workspace -D warnings`
chained to the commit in ONE `&&` chain (never a separate line), plus
`make check-matrix-anchors`, the full `run-interop-scan-plan.sh` chain green with the new leg, and
the nightly `scan-plan` suite green on the merged tip. Independent Critic must re-derive M-3
non-vacuity independently — that is the proof this whole unit turns on *(as shipped: the plan-shape
assert, not the span inequality — superseded framing, see task/lessons.md 2026-07-24)*.
