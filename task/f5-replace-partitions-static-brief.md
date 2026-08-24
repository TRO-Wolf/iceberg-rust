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

# F-5 — ReplacePartitions "static path" + multi-spec conflict (GAP_MATRIX row R104)

Branch `parity/f5-replace-partitions-static`, cut off main `4b3500e8f`.

## The nominal ask

> Close the static path (explicit partition tuple and row-filter forms) and the multi-spec conflict
> interop so R104 goes ✅.

## PHASE 0 — scope audit: HALF THE ASK IS VOID

**The "static path" half is void. There is no such API in Java 1.10.0 to port, and adding one would be
an ANTI-parity change.** Verified first-hand against the jars; every `javap` transcript is in the
ledger.

1. `org.apache.iceberg.ReplacePartitions` (iceberg-api 1.10.0) declares exactly **five** methods:
   `addFile`, `validateAppendOnly`, `validateFromSnapshot`, `validateNoConflictingDeletes`,
   `validateNoConflictingData`. No `replaceByRowFilter`, no explicit-partition selector.
   `BaseReplacePartitions` (iceberg-core 1.10.0) adds no public selector of its own.
2. `overwriteByRowFilter(Expression)` is a method of **`OverwriteFiles`**, and the fork already ships it
   (`crates/iceberg/src/transaction/overwrite_files.rs`), with a faithful per-spec residual +
   inclusive/strict-metrics resolution, partial-match rejection and
   `validateAddedFilesMatchOverwriteFilter`.
3. Spark decides which core action a static vs dynamic overwrite reaches, and it is **not** this action:
   `SparkWriteBuilder.overwrite(Filter[])` sets `overwriteByFilter` and `SparkWrite$OverwriteByFilter`
   calls `table.newOverwrite().overwriteByRowFilter(expr)`; only `overwriteDynamicPartitions()` /
   `SparkWrite$DynamicOverwrite` calls `table.newReplacePartitions()`.

So R104's residue clause "static `replaceByRowFilter`/explicit-partition APIs" asserted a Java API that
does not exist — the same defect class F-0 corrected on another row. **It is struck, with citations.**

## What the audit found INSTEAD

Auditing the multi-spec half turned up two real, silent divergences on the *existing* surface. Java
branches twice on `MergingSnapshotProducer.dataSpec()` — the spec shared by the **added** files — and the
fork had neither branch:

| Java site | Predicate | Java behavior | Fork behavior before this unit |
|---|---|---|---|
| `BaseReplacePartitions.apply` | `dataSpec().fields().isEmpty()` | adds `deleteByRowFilter(alwaysTrue)` ⇒ removes EVERY live data file, spec-agnostic | removed only files keyed `(data_spec_id, <tuple>)` ⇒ files under an older PARTITIONED spec survived |
| `BaseReplacePartitions.validate` | `dataSpec().isUnpartitioned()` | the `alwaysTrue` overloads of `validateAddedDataFiles` / `validateDeletedDataFiles` / `validateNoNewDeleteFiles` ⇒ any concurrent write anywhere conflicts | `PartitionSet`-keyed ⇒ a concurrent write under another spec was accepted |

Both are invisible on a single-spec table (every prior test and every interop leg), which is why they
survived three previous R104 increments. The apply-side one is a data-correctness divergence: the fork's
"full replace" was not full.

Note the predicates **differ on purpose** — `fields().isEmpty()` is strictly narrower than
`isUnpartitioned()` (= empty *or* all-VOID). An all-VOID spec is therefore wide for `validate` and narrow
for `apply`. That asymmetry is Java's; both halves are now pinned so nobody "harmonizes" them.

## Scope delivered

- Port both `dataSpec()` branches to Java's exact predicates (`replace_partitions.rs`).
- 8 mutation-proven unit tests on genuinely multi-spec fixtures (the module had **zero** before).
- Correct the R104 residue and record what actually remains.
- Update `crates/iceberg/src/transaction/map.md`.

## Explicitly NOT delivered (named residue, see the ledger)

- **The multi-spec conflict INTEROP leg.** No Java-oracle leg exists; adding one is a four-part change
  (an `InteropOracle` mode + a `run-interop-*.sh` driver + a Rust `interop_*.rs` consumer + a
  `SUITE_FLOOR_DEFAULT` ratchet). Deferred as its own unit.
- **`dataSpec()`'s two `Preconditions.checkState` throws** (no added files; added files spanning
  different specs). Porting them turns two currently-accepted commits into hard errors and would flip an
  existing pinned test.

## R104 verdict

**Stays 🟡.** The parity mandate requires an interop test before a row goes ✅, and the multi-spec leg —
the exact thing the residue names — does not exist. The unit closes the *implementation* gap and the
*unit-test* gap; the interop gap is untouched.
