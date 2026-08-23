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

# Evidence ledger — F-3: `RewriteDataFiles` composes `remove-dangling-deletes`

Companion to [the scope brief](f3-rewrite-dangling-deletes-brief.md). Dated 2026-08-23.

## 1. Bytecode citations (all re-decoded first-hand this session)

Tooling: `javap -p -c -constants -cp <jar> <class>`. Jars —
`API` = `/home/john/.m2/repository/org/apache/iceberg/iceberg-api/1.10.0/iceberg-api-1.10.0.jar`;
`RT` = `/home/john/CodeRepos/PrimarySync/fixtures/spark-mor-file-granularity/gen/.jars/iceberg-spark-runtime-4.0_2.13-1.10.0.jar`.

### 1.1 The option and its default — `API`, `org.apache.iceberg.actions.RewriteDataFiles`

```
  public static final java.lang.String REMOVE_DANGLING_DELETES = "remove-dangling-deletes";
  public static final boolean REMOVE_DANGLING_DELETES_DEFAULT = false;
```

### 1.2 The aggregate accessor SUMS the per-group values — `API`, `RewriteDataFiles$Result`

```
  public default int removedDeleteFilesCount();
    Code:
       0: aload_0
       1: invokeinterface #7,  1            // InterfaceMethod rewriteResults:()Ljava/util/List;
       6: invokeinterface #13,  1           // InterfaceMethod java/util/List.stream:()Ljava/util/stream/Stream;
      11: invokedynamic #53,  0             // InvokeDynamic #4:applyAsInt:()Ljava/util/function/ToIntFunction;
      16: invokeinterface #23,  2           // InterfaceMethod java/util/stream/Stream.mapToInt:(...)
      21: invokeinterface #29,  1           // InterfaceMethod java/util/stream/IntStream.sum:()I
      26: ireturn
```

### 1.3 …but the per-group value is a CONSTANT ZERO — `API`, `RewriteDataFiles$FileGroupRewriteResult`

```
  public abstract org.apache.iceberg.actions.RewriteDataFiles$FileGroupInfo info();
  public abstract int addedDataFilesCount();
  public abstract int rewrittenDataFilesCount();
  public default long rewrittenBytesCount();
    Code:
       0: lconst_0
       1: lreturn
  public default int removedDeleteFilesCount();
    Code:
       0: iconst_0
       1: ireturn
```

Three corroborating decodes, all from `RT`:

- `BaseRewriteDataFiles$FileGroupRewriteResult.removedDeleteFilesCount()` is a `default` whose body is
  `aload_0; invokespecial RewriteDataFiles$FileGroupRewriteResult.removedDeleteFilesCount:()I; ireturn`
  — it re-delegates to the constant-zero default, adding nothing.
- `ImmutableRewriteDataFiles$FileGroupRewriteResult` carries a
  `private int removedDeleteFilesCountInitialize()` whose body is
  `invokespecial BaseRewriteDataFiles$FileGroupRewriteResult.removedDeleteFilesCount:()I` — i.e. the
  Immutables `@Value.Default` initializer is that same zero.
- `grep -n removedDeleteFilesCount` over the FULL `javap -c` of
  `org.apache.iceberg.spark.actions.RewriteDataFilesSparkAction` (994 lines) returns **exactly one**
  line — offset 156 in `execute()`, the top-level read. Population: 1 of 1 occurrences in that class;
  nothing on the `RewriteDataFiles` path ever calls the per-group builder setter.

**Shape ruling:** the sum is identically `0`, so the fork adds the field at the TOP level only and
`FileGroupRewriteResult` gains nothing. Adding a per-group field would invent a shape Java does not
populate.

### 1.4 Placement, the early returns, and the fold — `RT`, `RewriteDataFilesSparkAction.execute()`

```
       4: getfield      #147  // Field table:Lorg/apache/iceberg/Table;
       9: invokeinterface #217  // Table.currentSnapshot:()Lorg/apache/iceberg/Snapshot;
      12: getstatic     #219  // Field EMPTY_RESULT   <-- early return #1 (no current snapshot)
      15: areturn
      ...
      44: aload_3
      45: invokevirtual #241  // FileRewritePlan.totalGroupCount:()I
      48: ifne          74
      70: getstatic     #219  // Field EMPTY_RESULT   <-- early return #2 (empty plan)
      73: areturn
      ...
     113: aload_0
     114: getfield      #273  // Field removeDanglingDeletes:Z
     117: ifeq          166
     120: new           #275  // class ...RemoveDanglingDeletesSparkAction
     129: getfield      #147  // Field table:Lorg/apache/iceberg/Table;
     132: invokespecial #276  // RemoveDanglingDeletesSparkAction."<init>":(SparkSession;Table;)V
     139: invokevirtual #279  // RemoveDanglingDeletesSparkAction.execute:()...RemoveDanglingDeleteFiles$Result;
     142: invokeinterface #283 // RemoveDanglingDeleteFiles$Result.removedDeleteFiles:()Ljava/lang/Iterable;
     147: invokestatic  #289  // Iterables.size:(Ljava/lang/Iterable;)I
     152: aload         5
     156: invokevirtual #292  // ImmutableRewriteDataFiles$Result.removedDeleteFilesCount:()I
     161: iadd
     162: invokevirtual #296  // ImmutableRewriteDataFiles$Result.withRemovedDeleteFilesCount:(I)...
     165: areturn
     166: aload         5
     168: areturn
```

Four rulings drop straight out:

1. **Empty plan ⇒ nothing runs.** Both `EMPTY_RESULT` returns are before offset 113.
2. **Non-empty plan + flag ⇒ it runs, even if it then removes nothing.** The only guard at 117 is the
   flag; there is no "removed something" precondition. Whether a snapshot is committed is the
   sub-action's own decision — `RemoveDanglingDeletesSparkAction.doExecute` (`RT`) calls
   `commit(...)` at offset 93 only after `DeleteFileSet.isEmpty()` is false at offsets 86-90.
3. **A failure fails the whole action.** `javap` prints **no `Exception table`** for `execute()`
   (checked by extracting the method's full listing and grepping for `Exception table` — 0 hits), so
   the sub-action's throw propagates.
4. **The table handle is CURRENT, so passing the final committed table is 1:1.**
   `RewriteDataFilesSparkAction.commitManager(long)` constructs `RewriteDataFilesCommitManager` with
   `this.table` (offset 5: `getfield #147 table`), i.e. every group commit goes through the SAME
   handle, and `BaseMetastoreTableOperations.commit` (`RT`) calls `requestRefresh()` at offset 83
   right after `doCommit`. `RemoveDanglingDeletesSparkAction` itself does no refresh — its ctor only
   stores the table (offsets 0-10) and `findDanglingDeletes` reads `loadMetadataTable(table, ENTRIES)`
   — it does not need to, because the handle it is given is already post-commit.

## 2. What changed

All production changes are in `crates/iceberg/src/maintenance/rewrite_data_files.rs`.

| Where | What |
|---|---|
| module docs, `# The composed remove-dangling-deletes sub-action` | New section carrying the offsets above, the empty-plan exemption, the failure posture, and the table-handle ruling incl. the named residue |
| module docs, `# Defaults (Java parity)` | `remove_dangling_deletes = false` added to the list |
| module docs, `# Empty plan` | Sentence added: the sub-action does not run on an empty plan |
| `use crate::maintenance::RemoveDanglingDeleteFiles;` | New import |
| `RewriteDataFilesResult` | `#[non_exhaustive]` + new `pub removed_delete_files_count: usize` with the shape rationale from §1.3 |
| `RewriteDataFiles` struct + `new()` | New `remove_dangling_deletes: bool` field, initialised `false` |
| `RewriteDataFiles::remove_dangling_deletes(bool)` | New builder method |
| `RewriteDataFiles::execute` step 7 | `if self.remove_dangling_deletes { let removed = RemoveDanglingDeleteFiles::new(table).execute(catalog).await?; result.removed_delete_files_count += removed.removed_delete_files.len(); }` — after the loop, after both early returns |

Tests (same file, `mod tests`): helpers `live_delete_file_paths`, `remove_data_files`,
`dangling_after_compaction_fixture`, plus four `#[tokio::test]`s (§4).

`FileGroupRewriteResult` is unchanged. `remove_dangling_delete_files.rs` is unchanged (it was
mutated and restored during §4; `git status` confirms it is identical to HEAD).

## 3. Compatibility measurement — MEASURED, not assumed

`RewriteDataFilesResult` is `pub` with all-`pub` fields, so adding a field is breaking for any
cross-crate exhaustive struct literal. Sibling unit F-2 set the precedent of adding
`#[non_exhaustive]` after measuring that the consuming engine never struct-literals the type. The
same measurement, run this session:

Consumer working copies located (both are RePark engine trees on this machine):

```
$ grep -rIln "iceberg::maintenance" /home/john/CodeRepos --include=*.rs | grep -v openSource/apacheIcebergRust | sed 's#/[^/]*$##' | sort -u
/home/john/CodeRepos/BigRustSparkRebuild/crates/repark-sql/src
/home/john/CodeRepos/LocalRepark/repark/crates/repark-spark/src
```

Struct-literal probe (the breaking pattern), whole of `/home/john/CodeRepos` minus the fork:

```
$ grep -rIn "RewriteDataFilesResult\s*{\|FileGroupRewriteResult\s*{" /home/john/CodeRepos --include=*.rs | grep -v openSource/apacheIcebergRust
EXIT:1        # no output — zero matches
```

Type-name probe over the same two consumer trees:

```
$ grep -rIn "RewriteDataFilesResult|FileGroupRewriteResult" <both trees> --include=*.rs
(no matches — neither type is even NAMED)
$ grep -rIn "RewriteDataFiles" /home/john/CodeRepos/LocalRepark/repark --include=*.rs
crates/repark-spark/src/call.rs:77:  use iceberg::maintenance::{DeleteOrphanFiles, RewriteDataFiles, RewritePositionDeleteFiles};
crates/repark-spark/src/call.rs:805: let result = RewriteDataFiles::new(table).execute(...)
$ grep -rIn "RewriteDataFiles" /home/john/CodeRepos/BigRustSparkRebuild --include=*.rs
crates/repark-sql/src/call.rs:44:  use iceberg::maintenance::RewriteDataFiles;
crates/repark-sql/src/call.rs:620: let result = RewriteDataFiles::new(table).execute(...)
```

Both consumers only READ fields off the returned value (`result.rewritten_data_files_count`,
`result.added_data_files_count`, `result.rewritten_bytes_count`). Population: 2 of 2 located consumer
trees, 0 struct literals of either type, 0 mentions of either type name.

**Ruling:** `#[non_exhaustive]` is added to `RewriteDataFilesResult` only (the type that gained a
field). It is a **consumed-surface change** and must be named as such in the PR body: from outside
the `iceberg` crate the struct can no longer be built with a struct literal or exhaustively
destructured. Measured impact on the located consumers: none. `FileGroupRewriteResult` is left
without `#[non_exhaustive]` — it gained no field, and adding the attribute there would be a gratuitous
breaking change.

**Directly relevant find.** `LocalRepark/repark/crates/repark-spark/src/call.rs:820-838` hard-codes
the `removed_delete_files_count` output column to `0` with a comment naming
`REMOVE_DANGLING_DELETES_DEFAULT` — precisely the handoff item this unit unblocks. That call site is
NOT changed here (it lives in another repo). The same comment records a live-Spark observation worth
carrying forward: on a **v3** table the Spark oracle reported `removed_delete_files_count = 6` with
NO option set, because deletion vectors are file-scoped and die with the data file they reference.
That is a v3 behaviour this unit does not model — see residue R-3.

## 4. Mutation proofs — every one EXECUTED, observed RED, and reverted

Method: a script (`/tmp/f3_mut.py`) that asserts the target text occurs **exactly once** and
**hard-fails (exit 2) if it cannot be applied** — a sabotage that did not corrupt anything proves
nothing — backs the file up, applies the mutation, runs the single named test, and restores the file
from the backup in a `finally` block. Verbatim harness output is reproduced below.

| # | Mutation | Test | Result |
|---|---|---|---|
| M1 | `remove_dangling_deletes: false` → `true` in `new()` (default flipped ON) | `test_remove_dangling_deletes_defaults_off` | **RED**, exit 101 |
| M2 | `if self.remove_dangling_deletes {` → `if false && self.remove_dangling_deletes {` (step 7 never runs) | `test_remove_dangling_deletes_on_removes_the_dangling_delete` | **RED**, exit 101 |
| M3 | empty-plan early return re-written to run the dangling step BEFORE returning (Java's ordering inverted) | `test_empty_plan_skips_the_dangling_step_entirely` | **RED**, exit 101 |
| M4 | in `remove_dangling_delete_files.rs`: position-delete clause `seq < min` → `seq <= min` | `test_remove_dangling_deletes_on_with_nothing_dangling_commits_no_snapshot` | **RED**, exit 101 |
| M5 | the fold `+= removed.removed_delete_files.len()` → `+= 0 * ...len()` (removal happens, count lies) | `test_remove_dangling_deletes_on_removes_the_dangling_delete` | **RED**, exit 101 |
| M6 | in `remove_dangling_delete_files.rs`: drop `transaction.commit(catalog).await?` (removals reported but never committed) | `test_remove_dangling_deletes_on_removes_the_dangling_delete` | **RED**, exit 101 |

Every run above was executed against the tree as committed (after `cargo fmt`), so the panic line
numbers below resolve in the delivered file — with ONE stated exception: **M3 inserts 7 lines into
`execute()`**, so its reported `3262` is the MUTATED tree; the same assertion sits at **3255** in the
committed file (`assert_eq!(result, RewriteDataFilesResult::default(), ...)`, verified by
`sed -n '3253,3258p'`). No other mutation changes the line count of `rewrite_data_files.rs`
(M1/M2/M5 are one-line-for-one-line; M4/M6 touch a different file).

Observed failures, verbatim:

```
=== M1 exit=101 ===
test ...::test_remove_dangling_deletes_defaults_off ... FAILED
panicked at crates/iceberg/src/maintenance/rewrite_data_files.rs:3081:9:
assertion `left == right` failed: the sub-action did not run, so nothing was removed
  left: 1
 right: 0

=== M2 exit=101 ===
test ...::test_remove_dangling_deletes_on_removes_the_dangling_delete ... FAILED
panicked at crates/iceberg/src/maintenance/rewrite_data_files.rs:3124:9:
assertion `left == right` failed: the one dangling delete file was removed (population: the table's 1 delete file)
  left: 0
 right: 1

=== M3 exit=101 ===
test ...::test_empty_plan_skips_the_dangling_step_entirely ... FAILED
panicked at crates/iceberg/src/maintenance/rewrite_data_files.rs:3262:9:
assertion `left == right` failed: an empty plan returns a zero-count result even with the flag on
  left: RewriteDataFilesResult { added_data_files_count: 0, rewritten_data_files_count: 0, rewritten_bytes_count: 0, removed_delete_files_count: 1, file_groups: [] }
 right: RewriteDataFilesResult { added_data_files_count: 0, rewritten_data_files_count: 0, rewritten_bytes_count: 0, removed_delete_files_count: 0, file_groups: [] }

=== M4 exit=101 ===
test ...::test_remove_dangling_deletes_on_with_nothing_dangling_commits_no_snapshot ... FAILED
panicked at crates/iceberg/src/maintenance/rewrite_data_files.rs:3191:9:
assertion `left == right` failed: nothing dangled by Java's predicate (population: the table's 1 delete file)
  left: 1
 right: 0

=== M5 exit=101 ===
test ...::test_remove_dangling_deletes_on_removes_the_dangling_delete ... FAILED
panicked at crates/iceberg/src/maintenance/rewrite_data_files.rs:3124:9:
assertion `left == right` failed: the one dangling delete file was removed (population: the table's 1 delete file)
  left: 0
 right: 1

=== M6 exit=101 ===
test ...::test_remove_dangling_deletes_on_removes_the_dangling_delete ... FAILED
panicked at crates/iceberg/src/maintenance/rewrite_data_files.rs:3130:9:
```

### 4.1 Anti-vacuity accounting (which assertion each mutation actually killed)

- **M5 vs M2.** M5 leaves the removal happening and only corrupts the arithmetic; the test still dies
  at line 3124, the `removed_delete_files_count == 1` assertion. That proves the count assertion is
  **not dominated** by the `rewritten_data_files_count == 6` assertion two lines above it.
- **M6 vs M5.** M6 keeps the count honest (the sub-action still returns the file it decided to
  remove) but never commits, so the test dies one assertion LATER, at line 3130 — the
  `live_delete_file_paths(...).is_empty()` check. That is the only mutation that isolates the
  on-disk assertion from the in-memory count, and it is the reason M6 exists: under M2 and M5 the
  count assertion fires first and the on-disk assertion is never reached.
- **M4** is the delete-corruption edge. It flips Java's STRICT `<` position-delete clause to `<=`,
  which makes the still-applicable same-sequence delete look dangling. The "nothing dangling" test
  catches it, so that test is pinning Java's exact predicate rather than merely observing a zero.
- **Disclosed domination that remains.** In
  `test_remove_dangling_deletes_on_removes_the_dangling_delete` the snapshot-count assertion
  (`snapshots_before + 2`) is reached only when both earlier assertions pass; no mutation in this set
  kills it in isolation. Its twin in
  `test_remove_dangling_deletes_on_with_nothing_dangling_commits_no_snapshot`
  (`snapshots_before + 1`, the "no empty GC snapshot" invariant) IS killed in isolation by M4, in a
  fixture where the count and delete-set assertions fire first — so under M4 that test dies at the
  count. Strictly: the "+1 / no empty snapshot" line is itself dominated in its own test. It is kept
  as documentation of the Java rule (`RemoveDanglingDeletesSparkAction.doExecute` commits only when
  the dangling set is non-empty) rather than claimed as independently proven.

### 4.2 A stale-artifact hazard in the harness, and why it did not corrupt any result

After the final revert, the first full `cargo test -p iceberg --lib` run reported **7 failures** — the
six pre-existing `remove_dangling_delete_files` tests plus one of the new ones — against a tree that
`git diff --exit-code` proved was byte-identical to HEAD for that file. Cause: `shutil.copyfile`
gives the `.f3bak` the mtime of the copy, and `shutil.move` restores that OLDER mtime onto the source,
so cargo saw a source older than the artifact built from the MUTATED text and skipped the rebuild.
`touch`ing both files and re-running gives `3388 passed; 0 failed; 1 ignored`.

Direction matters here. The stale artifact can only mislead in the direction it did — a false RED on
a clean tree — for the RUNS THAT MATTER, because every mutation run *writes* the file first and so
always carries a newer mtime than any artifact; each of M1-M6 genuinely recompiled. The hazard is
real for the post-revert verification, which is exactly where it fired and was caught. **Any future
mutation harness should `touch` the restored file** rather than trust `shutil.move`'s mtime.

## 5. Residue and divergence found, NOT fixed

- **R-1 — refresh window (named divergence, deliberate).** Java's `this.table` is refreshed from the
  CATALOG after each commit (`requestRefresh()`), so it would additionally observe a concurrent
  third-party commit landing between the last group commit and the dangling scan. This port hands the
  sub-action exactly the table its own last group commit produced. Narrower, never staler. Documented
  in the module docs; not emulated.
- **R-2 — the inherited resurrection race.** `RemoveDanglingDeleteFiles` commits with no
  concurrent-conflict validation (both Java's `BaseRewriteFiles.validate` and the fork's skip the
  check when the replaced-DATA set is empty). Composing it into `RewriteDataFiles` does not create
  that race, but it does put it on a new path. Already documented at length on
  `remove_dangling_delete_files.rs`; unchanged here.
- **R-3 — v3 deletion vectors.** RePark's live Spark 4.0.1 + Iceberg 1.10.0 oracle recorded
  `removed_delete_files_count = 6` on a **v3** table with the option OFF, because a DV dies with the
  data file it references. This unit models only the opt-in sub-action; it makes no claim about what
  the count should be on v3, and the new tests are all `FormatVersion::V2`. Whoever admits v3 owns
  that column.
- **R-4 — no interop evidence.** Nothing here was round-tripped against Java. `R135` stays 🟡 for
  that reason among others.
- **R-5 — options-map surface.** Java configures this through a string-keyed options map
  (`REMOVE_DANGLING_DELETES`); the fork exposes a typed builder. The constant name is cited in the
  rustdoc so a future options-map layer can bind it, but no such layer is added.
- **R-6 — the consumer's hard-coded `0`.** `LocalRepark/repark/crates/repark-spark/src/call.rs:844`
  and `BigRustSparkRebuild/crates/repark-sql/src/call.rs` still emit a literal `0` for that column.
  Those are other repositories; unblocking them was the point of this unit, but wiring them is not
  part of it.

## 6. Process defect recorded honestly

Mid-unit, the first mutation cycle reverted with `git checkout -- <file>` on an **uncommitted** file
and destroyed the entire unit's work in progress. It was fully replayed from the edit scripts, and
every subsequent mutation used a file-copy backup restored in a `finally` block instead. Nothing was
lost from the delivered change, but the near-miss is the reason §4's harness never touches git. The
generalizable rule: **never use `git checkout --` as a mutation-revert while the change under test is
uncommitted** — the revert target and the work are the same file.
