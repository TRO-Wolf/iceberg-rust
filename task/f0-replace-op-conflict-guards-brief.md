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

# Scope brief — F-0: `Operation::Replace` missing from the two files-exist conflict guards

**Ledger id:** `F0-REPLACE-GUARDS-2026-08-23`
**Branch:** `parity/f0-replace-op-conflict-guards` (cut off `main` = `e69f7b0a`)
**Handoff item:** F-0 (P0), `~/.claude/plans/2026-08-23-opus-iceberg-rust-fork-handoff.md` §3
**Matrix rows in scope:** R105 (`DeleteFiles`), R106 (`RowDelta`) — status cells only.

## 1. The defect

`crates/iceberg/src/transaction/snapshot.rs` gates the "data files still exist" conflict
validation on two operation-set predicates. Both omit `Operation::Replace`:

| Fn (`snapshot.rs`) | Java set | Java members (bytecode) | Fork members | Gap |
|---|---|---|---|---|
| `operation_removes_data_files` (:1777) | `VALIDATE_DATA_FILES_EXIST_OPERATIONS` | `overwrite, replace, delete` | `Overwrite, Delete` | `Replace` |
| `operation_removes_data_files_skip_deletes` (:1791) | `VALIDATE_DATA_FILES_EXIST_SKIP_DELETE_OPERATIONS` | `overwrite, replace` | `Overwrite` | `Replace` |

## 2. Oracle — Java 1.10.0, bytecode-verified 2026-08-23

`javap -p -c org.apache.iceberg.MergingSnapshotProducer` over
`iceberg-core-1.10.0.jar` (Maven Central, sha to be recorded by the Actor), `static {}` block:

```
20: ldc_w  "overwrite" / 23: ldc_w "replace" / 26: ldc_w "delete"
29: ImmutableSet.of(3-arg) -> VALIDATE_DATA_FILES_EXIST_OPERATIONS
35: ldc_w  "overwrite" / 38: ldc_w "replace"
41: ImmutableSet.of(2-arg) -> VALIDATE_DATA_FILES_EXIST_SKIP_DELETE_OPERATIONS
```

The other three sets in the same `static {}` block were read at the same time and the fork
ALREADY matches all three — do not touch them:
`VALIDATE_ADDED_FILES_OPERATIONS = {append, overwrite}`,
`VALIDATE_ADDED_DELETE_FILES_OPERATIONS = {overwrite, delete}`,
`VALIDATE_ADDED_DVS_OPERATIONS = {overwrite, delete, replace}`.

## 3. Why the existing rationale is false

Both fns carry a doc rationale asserting `Operation::Replace` is unrepresentable and that
"Rust never records a `REPLACE` snapshot". **Both claims are false at `main`:**

- `spec/snapshot.rs:50` — `Replace` is a variant of `Operation`.
- `transaction/rewrite_files.rs:29` — "**Operation recorded:** always `Operation::Replace`".
- `RewriteDataFiles`, `RemoveDanglingDeleteFiles`, and `RewritePositionDeleteFiles` all commit
  through `RewriteFilesAction`, so the fork's OWN compaction path emits REPLACE snapshots.
- `operation_adds_dvs` (:1726), **14 lines above the first defect**, already includes
  `Operation::Replace` and its doc explicitly says so, citing the same Java class. The file
  contradicts itself.

## 4. Consequence (the corruption line)

A `RowDelta` / `DeleteFiles` commit whose files-exist validation is enabled does NOT inspect a
concurrently-committed REPLACE (compaction) snapshot. Compaction writes `Deleted` tombstones for
the data files it rewrote; the validation walks for exactly those tombstones and skips them
because of the op filter. The commit therefore SUCCEEDS, and the position deletes it carries
reference data files that no longer exist — the rows they deleted are live again in the
compacted output. Silent, no error, no retry. Armed unconditionally whenever compaction runs
concurrently with merge-on-read writes, i.e. the normal steady state of a maintained table.

## 5. The change

1. Add `Operation::Replace` to both predicates.
2. Replace the false rationale at :1770-1773 and :1788-1789 with the bytecode-verified truth,
   and correct the two downstream restatements at :2267 and :2399.
3. Tests (see §6).
4. Flip the R105 / R106 status cells to record the fix, dated. Run `make check-matrix-anchors`.

**Do NOT** touch `docs/parity/archive/2026-06_matrix-cell-narratives.md` — dated archives are
historical epochs by CLAUDE.md's working conventions, even where superseded.

## 6. Test duty — the part that actually matters

The existing suite is GREEN against the defect. It was built around the op set that IS covered,
so coverage of `{Overwrite, Delete}` reads as coverage of the predicate. Required:

- **One test per predicate**, each proving a concurrent **REPLACE** snapshot is now inspected.
  These must be the tests that go RED when `Operation::Replace` is removed from that ONE
  predicate — verify each mutation INDIVIDUALLY, not as a batch, and record which tests reddened
  and the population (`N of M`).
- **The corruption-level test**: a real concurrent compaction (`RewriteFiles`/`RewriteDataFiles`
  producing a REPLACE snapshot) between transaction build and commit, with a `RowDelta` carrying
  position deletes over a rewritten data file. Assert the commit is REJECTED, non-retryable,
  naming the missing file. Under the defect this test asserts a successful commit and live
  resurrected rows — state in the ledger which it does at base.
- **The `skip_deletes` twin must not be forgotten**: `skip_deletes == true` (RowDelta's default)
  and `== false` (DeleteFiles) route to DIFFERENT predicates. One chain per arm.
- Confirm the gate actually RUNS the new tests (`cargo test -p iceberg --lib` reaches them).

## 7. Out of scope — name, do not fix

- The `RewriteDataFiles` equal-arity spec-evolution partition wrong-stamp
  (`rewrite_data_files.rs:671-683`) — a separate fork-found defect, its own unit.
- The inert `validate_from_snapshot` on delete-only maintenance actions
  (`rewrite_files.rs:587-589` early-returns when `deleted_data_files.is_empty()`) — related,
  separate unit. Note it in the ledger as a named residue.
- Any engine-side change. F-2 and the rest of the handoff queue.

## 8. Repo law

`AGENTS.md` + `CLAUDE.md` govern. No `Cargo.toml`/`Cargo.lock` edits. No `git add -A`.
Verification gate chained to the commit in ONE `&&` chain. Every test asserts; `.expect(ctx)`
over `.unwrap()`. Update any `map.md` this change makes inaccurate, in the same change.
