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

# Scope brief — F-3: `RewriteDataFiles` composes `remove-dangling-deletes`

**Unit:** engine handoff item F-3 (P2). **Branch:** `parity/f3-rewrite-dangling-deletes`, cut off main
`5e7b2e4f8`. **Matrix row:** [`R135`](../docs/parity/GAP_MATRIX.md) (`RewriteDataFiles`) — NARROWED,
**not** flipped; it stays 🟡.

## The problem

`RewriteDataFiles` had no `remove-dangling-deletes` option, so the standalone
`RemoveDanglingDeleteFiles` sub-action (row `R137`, ✅) could not be composed into a compaction run.
A consuming engine exposing Spark's `rewrite_data_files` result shape had to hard-code
`removed_delete_files_count` to `0`, with a comment explaining that the non-default path was
unreachable rather than that the number was measured.

## The Java 1.10.0 oracle

Jars: `~/.m2/repository/org/apache/iceberg/iceberg-api/1.10.0/iceberg-api-1.10.0.jar` and
`.../fixtures/spark-mor-file-granularity/gen/.jars/iceberg-spark-runtime-4.0_2.13-1.10.0.jar`.
Every citation below is re-decoded first-hand in the ledger.

- `RewriteDataFiles.REMOVE_DANGLING_DELETES = "remove-dangling-deletes"`,
  `REMOVE_DANGLING_DELETES_DEFAULT = false` (api bytecode constant pool).
- `RewriteDataFiles$Result.removedDeleteFilesCount()` is a `default` that SUMS the per-group values;
  `RewriteDataFiles$FileGroupRewriteResult.removedDeleteFilesCount()` is a `default` whose whole body
  is `iconst_0; ireturn`. The per-group value is never set on the `RewriteDataFiles` path, so the sum
  is identically `0` and the entire non-zero contribution is
  `ImmutableRewriteDataFiles$Result.withRemovedDeleteFilesCount(existing + n)` at the TOP level.
- `RewriteDataFilesSparkAction.execute()`: `EMPTY_RESULT` returns at offsets 15
  (`currentSnapshot() == null`) and 73 (`totalGroupCount() == 0`) BOTH precede the dangling step at
  offset 113; the step is `if (removeDanglingDeletes) { n = Iterables.size(new
  RemoveDanglingDeletesSparkAction(spark, this.table).execute().removedDeleteFiles());
  result.withRemovedDeleteFilesCount(result.removedDeleteFilesCount() + n); }`. No exception table
  covers it.
- `this.table` is the handle the commit manager already committed through, and every
  `TableOperations.commit` implementation leaves it CURRENT — `BaseMetastoreTableOperations.commit`
  calls `requestRefresh()` (offset 83) after `doCommit`, `HadoopTableOperations.commit` sets
  `shouldRefresh` (offset 245), and `RESTTableOperations.commit` calls `updateCurrentMetadata`
  (offset 262), so the conclusion is not metastore-specific. Passing the loop's final committed table is 1:1, not a divergence.

## The ask (delivered)

1. Opt-in builder `RewriteDataFiles::remove_dangling_deletes(bool)`, default `false`.
2. After the group loop, when the flag is set AND the plan was non-empty, run
   `RemoveDanglingDeleteFiles` against the table state the rewrite left behind and fold its removal
   total into the result. Errors propagate (Java-identical).
3. `RewriteDataFilesResult::removed_delete_files_count` — top level ONLY, mirroring the decoded
   Java shape. `FileGroupRewriteResult` is untouched.
4. `R135` cell narrowed; still 🟡.

## Explicitly OUT of scope

- Flipping `R135` to ✅ — the interop deferral and every other deferral in the cell remain.
- Java interop evidence for this path (no Spark round-trip was run; the oracle is bytecode only).
- A per-group `removed_delete_files_count` field — Java never populates one (see the oracle).
- `partial-progress.*`, concurrency, sort/z-order, delete-ratio, `output-spec-id`, `rewrite_all`,
  `rewrite-job-order`, input-file splitting — all still deferred, untouched by this unit.
- An options-map / string-keyed configuration surface: the fork's builder is typed.
- Any change to `RemoveDanglingDeleteFiles` itself (row `R137`) — it is consumed unchanged.
