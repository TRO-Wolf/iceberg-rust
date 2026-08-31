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

# F-6 — branch commit target (`to_branch`, REF-1)

Branch `parity/f6-branch-commit-target`, cut off main `d4f55e1d` (independent of F-7).

GAP_MATRIX: promote `SnapshotUpdate.toBranch` out of row R156 into new row R168.

## The nominal ask

Port Java `SnapshotUpdate.toBranch(String)` so every snapshot-producing action can commit onto a
named branch. Validation and retry must resolve against that branch head, not `main`.

WAP (`write.wap.enabled` / `stage_only`) is out of scope.

## PHASE 0 — producer set (Java 1.10.0 = 1.11.0 for this surface)

Measured against iceberg-api/core 1.10.0 jars and
`https://raw.githubusercontent.com/apache/iceberg/apache-iceberg-1.10.0/...`.
1.11.0 `SnapshotProducer.targetBranch` / `apply` / `commit` is the same contract; 1.11.0
`SnapshotUpdate` only adds unrelated `validateWith`.

| Java class | `toBranch` | Fork action |
|---|---|---|
| `FastAppend` | yes → `targetBranch(branch)` | `FastAppendAction` |
| `MergeAppend` | yes | `MergeAppendAction` |
| `BaseOverwriteFiles` | yes | `OverwriteFilesAction` |
| `BaseReplacePartitions` | yes | `ReplacePartitionsAction` |
| `BaseRewriteFiles` | yes | `RewriteFilesAction` |
| `BaseRowDelta` | yes | `RowDeltaAction` |
| `StreamingDelete` | yes | `DeleteFilesAction` |
| `BaseRewriteManifests` | no (interface default throws) | `RewriteManifestsAction` — not exposed |
| `CherryPickOperation` | no | `CherryPickAction` — not exposed |

`SnapshotUpdate.toBranch` default throws
`Cannot commit to branch %s: %s does not support branch commits`.

### Java semantics (measured, not assumed)

1. **Default target is `main`.** `SnapshotProducer` field `targetBranch = SnapshotRef.MAIN_BRANCH`.
2. **Null branch is rejected** at setter time: `Invalid branch name: null`.
3. **A tag is rejected:** `{name} is a tag, not a branch. Tags cannot be targets for producing snapshots`.
4. **A missing branch is allowed and is created at commit.** `TableMetadata.Builder.setBranchSnapshotInternal`
   builds `SnapshotRef.branchBuilder(id)` when the ref is absent. Java test
   `TestFastAppend.testAppendCreatesBranchIfNeeded`.
5. **Parent snapshot is `SnapshotUtil.latestSnapshot(base, targetBranch)`:**
   - `main` (or null) → `currentSnapshot()`
   - existing branch → that ref's snapshot
   - **missing branch → `currentSnapshot()`** (the new branch forks from current `main`, and
     `main` does not move)
6. **Summary previous head is `previous.ref(targetBranch)`, not `latestSnapshot`.** A brand-new
   branch seeds totals from zero.
7. **OCC requirement** is `AssertRefSnapshotID` on the named ref. Missing ref → assert the ref
   does not exist (`snapshot-id` null). Retry refreshes, then re-resolves the named branch.
8. **Empty table `toBranch`:** `currentSnapshot` stays null; the branch is created. Java
   `testAppendToBranchEmptyTable`.

## Catastrophe class

Wrong-target commit: `to_branch("audit")` advances `main`, or validation/retry reads `main`
while the named branch moves.

Pins:

- Commit lands only on the named ref. Other refs' heads (including `main`) are byte-stable.
- Concurrent-commit retry re-resolves the named branch.

## Explicitly NOT delivered

- WAP / `stage_only` combined with `to_branch` (named residue; `stage_only` still emits no
  `SetSnapshotRef`).
- `to_branch` on `RewriteManifests` / `CherryPick` (Java throws; not exposed).
- Bidirectional Java interop (unit-proven only; row R168 stays 🟡).
- Engine DML routing (engine-side after this lands).
