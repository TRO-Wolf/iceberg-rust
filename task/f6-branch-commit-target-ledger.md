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

# F-6 evidence ledger — branch commit target (row R168)

Charter: [f6-branch-commit-target-brief.md](f6-branch-commit-target-brief.md).

## 1. Java 1.10.0 / 1.11.0 evidence

Sources read 2026-08-30 (read-only):

- `/home/john/.m2/repository/org/apache/iceberg/iceberg-{api,core}/1.10.0/`
- `https://raw.githubusercontent.com/apache/iceberg/apache-iceberg-1.10.0/core/src/main/java/org/apache/iceberg/SnapshotProducer.java`
- same tag: `SnapshotUpdate.java`, `SnapshotUtil.java`, `TableMetadata.java`, `FastAppend.java`,
  `MergeAppend.java`, `BaseOverwriteFiles.java`, `BaseReplacePartitions.java`,
  `BaseRewriteFiles.java`, `BaseRowDelta.java`, `StreamingDelete.java`,
  `BaseRewriteManifests.java`, `CherryPickOperation.java`, `MergingSnapshotProducer.java`,
  `TestFastAppend.java`, `TestTransaction.java`
- 1.11.0 `SnapshotProducer.java` / `SnapshotUpdate.java` (target-branch contract unchanged)

### 1a. `SnapshotUpdate.toBranch` default throws

```
default ThisT toBranch(String branch) {
  throw new UnsupportedOperationException(
      String.format(
          "Cannot commit to branch %s: %s does not support branch commits",
          branch, this.getClass().getName()));
}
```

`BaseRewriteManifests` and `CherryPickOperation` do not override it.

### 1b. `SnapshotProducer.targetBranch(String)`

```
Preconditions.checkArgument(branch != null, "Invalid branch name: null");
boolean refExists = base.ref(branch) != null;
Preconditions.checkArgument(
    !refExists || base.ref(branch).isBranch(),
    "%s is a tag, not a branch. Tags cannot be targets for producing snapshots",
    branch);
this.targetBranch = branch;
```

Missing refs pass. Tags fail. Null fails.

### 1c. `SnapshotUtil.latestSnapshot(TableMetadata, String)`

```
if (branch == null || branch.equals(SnapshotRef.MAIN_BRANCH)) {
  return metadata.currentSnapshot();
}
SnapshotRef ref = metadata.ref(branch);
if (ref == null) {
  return metadata.currentSnapshot();
}
return metadata.snapshot(ref.snapshotId());
```

A missing branch parents off current `main`. `main` does not move.

### 1d. `commit()` writes the named ref

```
if (stageOnly) {
  update.addSnapshot(newSnapshot);
} else {
  update.setBranchSnapshot(newSnapshot, targetBranch);
}
```

`setBranchSnapshotInternal` creates the branch when the ref is absent.

### 1e. Java unit pins (TestFastAppend)

- `testAppendToExistingBranch` — main stays at 1, branch moves to 2
- `testAppendCreatesBranchIfNeeded` — missing branch is created; main stays
- `testAppendToBranchEmptyTable` — current snapshot stays null
- `testAppendToNullBranchFails` — `Invalid branch name: null`
- `testAppendToTagFails` — tag message above

## 2. Fork producer partition (quantified clause)

Every Java-supporting producer must expose `to_branch` and commit through
`SnapshotProducer` with that target:

1. `FastAppendAction`
2. `MergeAppendAction`
3. `OverwriteFilesAction`
4. `ReplacePartitionsAction`
5. `RewriteFilesAction`
6. `RowDeltaAction`
7. `DeleteFilesAction`

## 3. Clause pins (this unit)

| Clause | Pin |
|---|---|
| Existing branch, main byte-stable | `to_branch_existing_branch_does_not_move_main` |
| Missing branch is created | `to_branch_creates_missing_branch` |
| Empty table | `to_branch_empty_table_leaves_current_null` |
| Tag rejected | `to_branch_tag_is_rejected` |
| Other refs byte-stable | `to_branch_leaves_sibling_refs_byte_stable` |
| Retry re-resolves named branch | `to_branch_retry_resolves_named_branch_not_main` |
| Parent of new branch is current | `to_branch_new_branch_parents_off_current` |
| Per-producer domain | `every_snapshot_producer_commits_to_named_branch` |

## 4. Named residue

- WAP / `stage_only` + `to_branch` not requested.
- `RewriteManifests` / `CherryPick` keep Java's throwing default (not exposed).
- No Java interop leg in this unit (row R168 🟡).

## 5. Self Logic Review

Inputs: Java `toBranch` / `targetBranch` / `latestSnapshot` / `setBranchSnapshot`.
Outputs: `to_branch` on the seven producers; producer commit uses named ref;
validation/retry walk that ref.
Failure modes: wrong-target write; lost update on concurrent branch commit;
creating a branch that rewrites `main`; treating a tag as a branch.
Proceed.
