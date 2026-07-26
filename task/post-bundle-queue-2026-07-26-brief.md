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

# Post-bundle queue — signed off 2026-07-26

Continues [back-to-goal-2026-07-25-brief.md](back-to-goal-2026-07-25-brief.md) (Units 1-2 merged
#172/#173; SEPMO arc merged #174/#175; nightly interop fully green — the scan-plan arc's live
proof landed). Decisions, continuing the D-series:

| # | Decision (user, 2026-07-26) |
|---|---|
| D7 | Queue order signed: **QA (R117 cross-task over-delete) → Unit 3 (breaking) → QB (delete-writer path bounds) → H7-S2 → H7-P1**, with QC slotted alongside Unit 3. |
| D8 | **QC approved** — the R161-completion `toHumanString` parity unit (format-visible: FIXED/BINARY partition-path segments move hex → Java base64; plus the `Transform::Unknown` taxonomy note and the `identity(float/double)` human-string divergence). D6-class approval granted. |

Cadence: Mode A per-unit PRs, merge-gated between units. Ladder per unit: max-effort Actor +
independent max-effort Critic (fresh context), remediation loop, push on CONVERGED per the
standing autonomy grant. All SEPMO v2.3 duties apply (R11-R13 are now canon: additive
contingencies, recorded dispositions, remand only explicitly).

---

## QA — R117: per-task DeleteFilter scope (the cross-task positional over-delete)

**Branch** `fix/delete-filter-per-task-scope` off `a08a0957`. **S1-class read correctness.**

**Defect (recorded in row R117, found while proving WG4b):** the per-scan `DeleteFilter` caches
parsed positional deletes keyed by the DATA-file path they name
(`caching_delete_file_loader.rs`'s upsert of `parse_positional_deletes_record_batch_stream`'s map
into the shared filter state), so a delete file loaded for ONE task contributes deletions to
ANOTHER task's data file. Reachable when a delete names a file outside its own partition/spec
bucket AND that file has ≥1 delete of its own (a task with no deletes never consults the cache —
why it stayed invisible). **Java builds one `DeleteFilter` per task over `task.deletes()` only** —
decode from 1.10.0 before coding. Recorded reproduction: the file-scoped-deletes interop fixture
with the control delete stamped `category=b` instead of the empty `category=c` wrongly deletes
id 30.

**Fix shape:** scope delete APPLICATION to each task's own delete set (`task.deletes()`), Java-
exact, while PRESERVING load/parse caching (re-parsing the same delete file per task would be a
performance regression, not parity — separate load-cache from apply-scope). Build on the G8
`PosDelState`/`EqDelState` machinery (merged #173) — do not regress the lost-wakeup/`Failed`
contracts (their pins must stay green untouched).

**Tests:** RED-first the recorded reproduction (control stamped `category=b` → id 30 must
SURVIVE post-fix; goes missing pre-fix); a two-task same-file control (a delete correctly named
by two tasks' sets still applies in both); the G8 concurrency pins untouched-green; equality-
delete path checked for the same class (the cell names positional — verify eq-deletes are
per-task already or fix in the same unit with its own pin); mutations: revert to shared-state
application → repro RED; over-scope (drop legitimate application) → control RED.

**Interop + matrix:** add the interop leg whose recipe the R117 cell records; if this closes the
DeleteFilter residue that holds R117 at 🟡, **re-promote 🟡→✅** with dated evidence (the Actor
adjudicates honestly — flip only if this was the sole blocker; anchors gate in the chain).

## Unit 3 — the breaking follow-up *(next after QA merges)*

Spec unchanged: [back-to-goal-2026-07-25-brief.md](back-to-goal-2026-07-25-brief.md) *Unit 3*
(`PartitionKey::new -> Result`, 58 sites / 34 files incl. 6 `no_run` doc fences +
`CurrentFileStatus` unwraps). Coordination duty: the PR body carries the full breaking-surface
list, and the RoadMapSync announcement warns RePark BEFORE their next repin (their Action E ack
expects it; their Group T/Y harness fixtures touch the constructor).

## QC — R161 completion: `toHumanString` parity *(alongside Unit 3; D8 APPROVED, format-visible)*

Fork renders FIXED/BINARY transform values in partition paths as HEX where Java 1.10.0 renders
**base64** (`Transform.toHumanString` → `TransformUtil.base64encode` for FIXED/BINARY — decode,
never assume); align byte-exactly, jar-oracle pinned across the tricky-value battery. Ride-alongs:
name `Transform::Unknown` in R161's byte-stability taxonomy; adjudicate + document (or align, if
Java-provable) the `identity(float/double)` human-string divergence. **Format-stability
attestation REQUIRED**: newly-written layouts for binary-typed partition values change to match
Java; existing tables stay readable (manifests authoritative) — same migration story as R161.
Update the R161 residue ledger to CLOSED items with dates; anchors gate.

## QB — R113-adjacent: delete-writer `file_path` bounds *(after Unit 3 per D7)*

Fork-written file-granularity position deletes are not self-identifying: parquet-rs truncates
byte-array statistics at 64 bytes by default and the metrics aggregator drops non-exact bounds,
so the read side's equal-bounds routing (the load-bearing leg for v2) never recognizes them
(Java's parquet-mr does not truncate row-group statistics). Investigate-first: exact-bounds for
the reserved `file_path` column (untruncated stats config for that column, or exact bounds
written through the metrics path another way) — no Cargo.toml changes without approval; if the
only clean fix needs one, STOP and report. Tests: a fork-written FILE-granularity delete must
route through `referenced_data_file_location`'s bounds leg (RED-first: today it does not);
round-trip vs the Java oracle. Matrix: R113 cell + the R117 residue line updated.

## H7-S2 → H7-P1 *(after QB)*

Scope home: [h7-dml-streaming-scope.md](h7-dml-streaming-scope.md). Re-scope at signing against
everything landed since (#124-era assumptions are stale; the P1 pre-condition — the
NOT-over-dropped-conjunct under-delete footgun — is unchanged and mandatory).

## Standing protocol

Worktree `iceberg-rust-ws`; claim before touch; content-verify merges (R4); chained gate per
commit (`typos . && cargo fmt --all -- --check && clippy -D warnings && iceberg lib + datafusion
all-targets && no-default-features` + anchors when the matrix moves); SEPMO v2.3 (S2 floor,
java-parity attestation per unit, format-stability attestation for QC, metrics ledger at
charter close); RoadMapSync announcements on every merge that moves RePark's pins.
