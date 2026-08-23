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

# Ledger — F-0: `Operation::Replace` added to the two files-exist conflict guards

**Ledger id:** `F0-REPLACE-GUARDS-2026-08-23`
**Branch:** `parity/f0-replace-op-conflict-guards` (cut off `main` = `e69f7b0a`)
**Scope:** [`task/f0-replace-op-conflict-guards-brief.md`](f0-replace-op-conflict-guards-brief.md)
**Matrix rows touched:** R105 (`DeleteFiles`), R106 (`RowDelta`) — status cells only.

## 1. Oracle — re-verified in this session

Artifact: `iceberg-core-1.10.0.jar`
**sha256 `54091489dbdcb31b5a4514372abfc908a7b0be69f76e785f1e82279be4fbd6cc`**

`javap -p -c -cp iceberg-core-1.10.0.jar org.apache.iceberg.MergingSnapshotProducer`, `static {}` block,
verbatim (all five sets, so the two in scope are read in their own context):

```
 8: ldc_w "append"  / 11: ldc_w "overwrite"
14: ImmutableSet.of(2-arg) -> VALIDATE_ADDED_FILES_OPERATIONS
20: ldc_w "overwrite" / 23: ldc_w "replace" / 26: ldc_w "delete"
29: ImmutableSet.of(3-arg) -> VALIDATE_DATA_FILES_EXIST_OPERATIONS
35: ldc_w "overwrite" / 38: ldc_w "replace"
41: ImmutableSet.of(2-arg) -> VALIDATE_DATA_FILES_EXIST_SKIP_DELETE_OPERATIONS
47: ldc_w "overwrite" / 50: ldc_w "delete"
53: ImmutableSet.of(2-arg) -> VALIDATE_ADDED_DELETE_FILES_OPERATIONS
59: ldc_w "overwrite" / 62: ldc_w "delete" / 65: ldc_w "replace"
68: ImmutableSet.of(3-arg) -> VALIDATE_ADDED_DVS_OPERATIONS
```

This matches the brief §2 oracle exactly. The other three sets were re-read at the same time and the
fork already matches all three — they were **not** touched.

## 2. The change

**`crates/iceberg/src/transaction/snapshot.rs`** — the two predicates:

| Fn | Before | After | Java set |
|---|---|---|---|
| `operation_removes_data_files` | `{Overwrite, Delete}` | `{Overwrite, Replace, Delete}` | `VALIDATE_DATA_FILES_EXIST_OPERATIONS` |
| `operation_removes_data_files_skip_deletes` | `{Overwrite}` | `{Overwrite, Replace}` | `VALIDATE_DATA_FILES_EXIST_SKIP_DELETE_OPERATIONS` |

**Doc sites corrected — six, not four.** The brief named four; two more restatements of the same
false claim were found in `row_delta.rs` while implementing and are corrected in the same change
(they became actively false the moment the predicates changed):

1. `snapshot.rs` — `operation_removes_data_files` rationale (the "`Replace` is unrepresentable / Rust
   never records a `REPLACE` snapshot" paragraph, replaced with the bytecode citation plus the
   `rewrite_files` counter-example).
2. `snapshot.rs` — `operation_removes_data_files_skip_deletes` rationale (same false claim).
3. `snapshot.rs` — `deleted_data_files_after` doc, the two-arm `skip_deletes` restatement + its
   "In BOTH cases the unrepresentable Java `REPLACE` operation is absent" sentence.
4. `snapshot.rs` — `validate_deleted_data_files` doc, both the op-set restatement and the
   "Conservative posture" paragraph's `REPLACE`-omission claim.
5. `row_delta.rs` — `RowDelta::validate` doc, check 3 (`{OVERWRITE}` / `{OVERWRITE, DELETE}` →
   `{OVERWRITE, REPLACE}` / `{OVERWRITE, REPLACE, DELETE}`).
6. `row_delta.rs` — the in-body comment at check 3 **and** the files-exist test-region banner, which
   restated the same two op sets.

A repo-wide grep for `unrepresentable` / ``never records a `REPLACE` `` / ``REPLACE` snapshot`` over
`crates/` and `docs/` (excluding dated archives, per the brief) shows no surviving instance of the
false claim; every remaining hit is an unrelated use of the word.

`docs/parity/archive/2026-06_matrix-cell-narratives.md` was **not** touched (brief §5 / the dated-archive
convention).

## 3. Tests

Two new tests, one per predicate, each an independent chain through a different action:

| Test | File | Action | `skip_deletes` | Predicate exercised |
|---|---|---|---|---|
| `test_row_delta_files_exist_rejects_concurrent_replace_compaction_of_referenced_file` | `transaction/row_delta.rs` | `row_delta()` (default, no `validate_deleted_files()`) | `true` | `operation_removes_data_files_skip_deletes` |
| `test_delete_files_exist_rejects_concurrent_replace_compaction_of_same_file` | `transaction/delete_files.rs` | `delete_files().validate_files_exist()` | `false` | `operation_removes_data_files` |

The row-delta test is the **corruption-level** test the brief required: a real concurrent compaction
(`tx.rewrite_files([f], [f-compacted])`, which commits `Operation::Replace`) lands between transaction
build and transaction commit, while the `RowDelta` carries a position delete over `f`.

Both tests **pin the fixture**: they assert the concurrent snapshot's
`summary().operation == Operation::Replace` before asserting the rejection, so neither can pass
vacuously through a different op-set member. Both assert the error kind (`DataInvalid`),
non-retryability, the `validateDataFilesExist` message, and the missing file's path. The
`delete_files` test additionally asserts the generic path-resolution message
(`"Missing required files to delete"`) is **absent** — without that assertion it would pass at base
by failing for the wrong reason (see §4).

**The gate runs them.** `cargo test -p iceberg --lib` selects them by name:

```
running 2 tests
test transaction::row_delta::tests::test_row_delta_files_exist_rejects_concurrent_replace_compaction_of_referenced_file ... ok
test transaction::delete_files::tests::test_delete_files_exist_rejects_concurrent_replace_compaction_of_same_file ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 3379 filtered out; finished in 0.04s
```

## 4. RED at base — verbatim

Both predicates reverted to their `main` bodies, new tests left in place, full lib suite:

```
failures:

---- transaction::delete_files::tests::test_delete_files_exist_rejects_concurrent_replace_compaction_of_same_file stdout ----

thread 'transaction::delete_files::tests::test_delete_files_exist_rejects_concurrent_replace_compaction_of_same_file' panicked at crates/iceberg/src/transaction/delete_files.rs:1314:9:
the error must be the validateDataFilesExist rejection, got: Missing required files to delete: test/a.parquet

---- transaction::row_delta::tests::test_row_delta_files_exist_rejects_concurrent_replace_compaction_of_referenced_file stdout ----

thread 'transaction::row_delta::tests::test_row_delta_files_exist_rejects_concurrent_replace_compaction_of_referenced_file' panicked at crates/iceberg/src/transaction/row_delta.rs:3305:45:
row delta must fail: a concurrent REPLACE (compaction) removed the referenced data file: Table { file_io: FileIO { ... } }

failures:
    transaction::delete_files::tests::test_delete_files_exist_rejects_concurrent_replace_compaction_of_same_file
    transaction::row_delta::tests::test_row_delta_files_exist_rejects_concurrent_replace_compaction_of_referenced_file

test result: FAILED. 3378 passed; 2 failed; 1 ignored; 0 measured; 0 filtered out; finished in 1.93s
```

**What each asserted at base — the answer the brief asked for:**

- **`row_delta` (the corruption-level test): the commit SUCCEEDED at base.** `expect_err` panicked
  because `tx.commit(&catalog)` returned `Ok(Table { .. })` — the panic payload is the committed
  `Table`. That is the defect in its literal form: a `RowDelta` carrying a position delete over
  `test/f.parquet` committed even though a concurrent `Operation::Replace` compaction had already
  removed `test/f.parquet` from the live set. No error, no retry — the position delete now references
  a data file that does not exist, and the rows it removed are live again in `test/f-compacted.parquet`.
- **`delete_files`: the commit failed at base, but for the WRONG reason.** The message was
  `Missing required files to delete: test/a.parquet` — the *generic path-resolution* check firing
  later in `commit`, not the `validateDataFilesExist` conflict guard. The conflict guard did not run
  at all. This is why the test asserts both the presence of the validation message and the absence of
  the path-resolution one; an assertion on "commit fails" alone would have been GREEN at base and
  proven nothing.

## 5. Per-predicate mutation results

**Population `M` = 3381 tests in `cargo test -p iceberg --lib`** (3380 executed + 1 `ignored`;
reported as `3378/3379 passed + N failed + 1 ignored + 0 filtered out` in the runs below). Each
mutation was applied and run **individually**; the union was never used to argue per-item necessity.
The harness hard-fails (exit 3) if the target text is absent, and restores from a pristine backup
before every mutation — no SKIP branch exists.

| # | Mutation | Result | Reddened |
|---|---|---|---|
| M0 | `Replace` removed from **both** predicates (= `main` source) | `FAILED. 3378 passed; 2 failed; 1 ignored` | **2 of 3381** — `test_row_delta_..._replace_compaction_of_referenced_file`, `test_delete_files_..._replace_compaction_of_same_file` |
| M1 | `Replace` removed from `operation_removes_data_files` ONLY | `FAILED. 3379 passed; 1 failed; 1 ignored` | **1 of 3381** — `test_delete_files_exist_rejects_concurrent_replace_compaction_of_same_file` |
| M2 | `Replace` removed from `operation_removes_data_files_skip_deletes` ONLY | `FAILED. 3379 passed; 1 failed; 1 ignored` | **1 of 3381** — `test_row_delta_files_exist_rejects_concurrent_replace_compaction_of_referenced_file` |

M1 and M2 redden **disjoint** single tests, which is the per-predicate necessity claim: the
`skip_deletes == true` and `skip_deletes == false` arms each have their own independent chain, and
coverage of one does not read as coverage of the other.

**Pre-existing suite was GREEN under both mutations** apart from the new tests — i.e. no test that
existed at `main` detects this defect on either arm. That is the concrete form of the brief's "the
existing suite is GREEN against the defect".

## 6. Consumed surface — does this change any public API the engine calls?

**No public API changes. Yes, an observable behavior change.** Stated explicitly for the PR body:

- **Signatures/visibility: unchanged.** `operation_removes_data_files` and
  `operation_removes_data_files_skip_deletes` are private `fn`s in
  `crates/iceberg/src/transaction/snapshot.rs`. `deleted_data_files_after` is `pub(crate)` and its
  signature is untouched. No `pub` item was added, removed, renamed, or re-typed. No trait, struct
  field, or error variant changed. Nothing downstream needs a re-pin to compile.
- **Behavior: a commit that previously succeeded can now be rejected.** Specifically, when a
  concurrent snapshot with `Operation::Replace` (i.e. any commit through
  `transaction::rewrite_files` — `RewriteDataFiles`, `RewritePositionDeleteFiles`,
  `RemoveDanglingDeleteFiles`) removed a data file that the committing operation requires, these
  paths now return a **non-retryable `ErrorKind::DataInvalid`** where they previously committed (or,
  for `DeleteFiles`, failed later with the different `"Missing required files to delete"` message):
  - `RowDelta` with `validate_data_files_exist(...)` enabled (`skip_deletes` arm, both settings),
  - `DeleteFiles` with `validate_files_exist()` enabled,
  - and, via the shared `deleted_data_files_after` walk with `skip_deletes == false`,
    `OverwriteFiles` / `ReplacePartitions` `validate_no_conflicting_deletes`-style checks now also
    inspect concurrent `Replace` snapshots.
- **The engine impact is the point, not a regression:** an engine running compaction concurrently
  with merge-on-read writes will start seeing these rejections instead of silently resurrecting rows.
  Engines that retry-on-conflict must treat this as a **non-retryable** validation failure and
  re-plan against the refreshed base (the error is `DataInvalid`, `retryable() == false`).
- **On-disk format: unchanged.** No encoding, no manifest shape, no metadata field is affected — this
  is a read-side gate over already-written tombstones.
- All checks remain **opt-in** exactly as before; nothing was turned on by default.

## 7. Named residues — out of scope, NOT fixed

Named here, not fixed, per brief §7:

1. **`RewriteDataFiles` equal-arity spec-evolution partition wrong-stamp** —
   `crates/iceberg/src/maintenance/rewrite_data_files.rs` (brief cites L671-683). A separate
   fork-found defect; its own unit.
2. **Inert `validate_from_snapshot` on delete-only maintenance actions** —
   `crates/iceberg/src/transaction/rewrite_files.rs` `validate` early-returns when
   `self.deleted_data_files.is_empty()`, so a rewrite that only removes DELETE files never runs the
   conflict validation its `validate_from_snapshot` implies. Related to this unit (same file, same
   validation family), separate unit.
3. **Engine-side changes and the rest of the F-queue** (F-2 onward) — untouched.

Additional residue found while implementing (new, not in the brief):

4. **No Java interop test.** This unit is unit-tested only; R105/R106 remain 🟡. The 1:1 evidence
   here is the bytecode oracle plus mutation-proved unit tests, not a byte-level round-trip. A
   Java-writes-compaction / Rust-rejects-row-delta interop pin would be the thing that upgrades this
   from "matches the decompiled set" to "matches Java's behavior in situ" — not attempted offline.
5. **The `deleted_data_files_after` walk still does not thread a conflict filter**, where Java's
   `validateDataFilesExist` accepts one. Pre-existing, documented in the fn's own docs as a
   conservative over-scan (can only over-reject). Untouched by this unit.

## 8. Gate

Chained to the commit in one `&&` chain:

```
typos . && cargo fmt --all -- --check && cargo clippy --all-targets --workspace -- -D warnings \
  && cargo test -p iceberg --lib && make check-matrix-anchors && git add <named paths> && git commit ...
```

Result recorded in the commit; see the session report for the pasted tail.
