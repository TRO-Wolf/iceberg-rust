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

# F-5 evidence ledger — ReplacePartitions `dataSpec()` scope branches (row R104)

Every claim below was executed. Charter: [f5-replace-partitions-static-brief.md](f5-replace-partitions-static-brief.md).

## 1. Bytecode evidence

All decodes run 2026-08-23 against
`/home/john/.m2/repository/org/apache/iceberg/iceberg-{api,core}/1.10.0/` and
`/home/john/CodeRepos/PrimarySync/fixtures/spark-mor-file-granularity/gen/.jars/iceberg-spark-runtime-4.0_2.13-1.10.0.jar`.

### 1a. `ReplacePartitions` has no static / row-filter API (refutes the old residue)

`javap -p -cp iceberg-api-1.10.0.jar org.apache.iceberg.ReplacePartitions`:

```
public interface org.apache.iceberg.ReplacePartitions extends org.apache.iceberg.SnapshotUpdate<org.apache.iceberg.ReplacePartitions> {
  public abstract org.apache.iceberg.ReplacePartitions addFile(org.apache.iceberg.DataFile);
  public abstract org.apache.iceberg.ReplacePartitions validateAppendOnly();
  public abstract org.apache.iceberg.ReplacePartitions validateFromSnapshot(long);
  public abstract org.apache.iceberg.ReplacePartitions validateNoConflictingDeletes();
  public abstract org.apache.iceberg.ReplacePartitions validateNoConflictingData();
}
```

`javap -p -cp iceberg-core-1.10.0.jar org.apache.iceberg.BaseReplacePartitions` adds only
`self()`, `operation()`, `toBranch(String)`, `validate(TableMetadata, Snapshot)`,
`apply(TableMetadata, Snapshot)`, `updateEvent()` and the inherited `SnapshotProducer` bridges
(`caseSensitive`, `set`, `commit`, `apply`, `deleteWith`, `scanManifestsWith`, `stageOnly`) — no
selector.

`overwriteByRowFilter` is on `OverwriteFiles` (`javap -p -cp iceberg-api-1.10.0.jar
org.apache.iceberg.OverwriteFiles` lists it first), and `BaseOverwriteFiles.overwriteByRowFilter`
decodes to a one-liner delegating to `ManifestFilterManager.deleteByRowFilter`:

```
  public org.apache.iceberg.OverwriteFiles overwriteByRowFilter(org.apache.iceberg.expressions.Expression);
       0: aload_0
       1: aload_1
       2: invokevirtual #52    // Method deleteByRowFilter:(Lorg/apache/iceberg/expressions/Expression;)V
       5: aload_0
       6: areturn
```

### 1b. Which core action Spark's static vs dynamic overwrite reaches

`javap -p -c` on the Spark runtime jar:

- `SparkWriteBuilder` `implements ... SupportsDynamicOverwrite, SupportsOverwrite`.
- `SparkWriteBuilder.overwriteDynamicPartitions()` sets `overwriteDynamic = true`.
- `SparkWriteBuilder.overwrite(Filter[])` converts via `SparkFilters.convert` and sets
  `overwriteByFilter = true` (unless the expression is `alwaysTrue` **and** `overwriteMode == "dynamic"`,
  in which case it sets `overwriteDynamic`).
- `SparkWriteBuilder$1.toBatch()` dispatches `overwriteByFilter ⇒ asOverwriteByFilter(overwriteExpr)`,
  `overwriteDynamic ⇒ asDynamicOverwrite()`.
- `SparkWrite$OverwriteByFilter`:
  `7: invokeinterface Table.newOverwrite:()` … `18: invokeinterface OverwriteFiles.overwriteByRowFilter:(…)`
- `SparkWrite$DynamicOverwrite`:
  `34: invokeinterface Table.newReplacePartitions:()` … `163: invokeinterface ReplacePartitions.addFile:(…)`

⇒ the "static path" is an `OverwriteFiles` concern, and it is already shipped.

### 1c. The two `dataSpec()` branches (what this unit ports)

`BaseReplacePartitions.apply`:

```
   0: aload_0
   1: invokevirtual #91   // Method dataSpec:()Lorg/apache/iceberg/PartitionSpec;
   4: invokevirtual #124  // Method org/apache/iceberg/PartitionSpec.fields:()Ljava/util/List;
   7: invokeinterface #128 // InterfaceMethod java/util/List.isEmpty:()Z
  12: ifeq 22
  16: invokestatic  #101  // Method org/apache/iceberg/expressions/Expressions.alwaysTrue:()
  19: invokevirtual #133  // Method deleteByRowFilter:(Lorg/apache/iceberg/expressions/Expression;)V
  22: … invokespecial MergingSnapshotProducer.apply
```

`BaseReplacePartitions.validate` (data branch at 0-47, deletes branch at 47-121):

```
   7: invokevirtual #91  // dataSpec()
  11: invokevirtual #95  // PartitionSpec.isUnpartitioned:()Z
  14: ifeq 33
  23: invokestatic  #101 // Expressions.alwaysTrue()
  27: invokevirtual #107 // validateAddedDataFiles:(TableMetadata;Long;Expression;Snapshot)V
  40: getfield      #44  // replacedPartitions:Lorg/apache/iceberg/util/PartitionSet;
  44: invokevirtual #111 // validateAddedDataFiles:(TableMetadata;Long;PartitionSet;Snapshot)V
```

with the identical `isUnpartitioned()` branch at 54-118 guarding `validateDeletedDataFiles` +
`validateNoNewDeleteFiles`.

Supporting decodes:

- `PartitionSet.contains(int, StructLike)` looks the **spec id** up in `partitionSetById` FIRST and
  returns `false` when that spec has no entry — so the partition-set form is genuinely spec-keyed.
- `PartitionSpec.isPartitioned()` = `fields.length > 0 && fields.stream().anyMatch(f -> !f.transform().isVoid())`,
  so `isUnpartitioned()` is true for an ALL-VOID spec while `fields().isEmpty()` is not. The two Java
  predicates are deliberately different.
- `MergingSnapshotProducer.dataSpec()` reads `newDataFilesBySpec.keySet()` and carries two
  `Preconditions.checkState` messages: "Cannot determine partition specs: no data files have been added"
  and "Cannot return a single partition spec: data files with different partition specs have been added".
- `BaseReplacePartitions.addFile` calls `dropPartition(file.specId(), file.partition())` and
  `replacedPartitions.add(file.specId(), file.partition())` — spec-keyed on both.

## 2. Code changed

All in `crates/iceberg/src/transaction/replace_partitions.rs`:

- new `enum ConflictScope { Partitions(HashSet<(i32, Struct)>), AllFiles }` + `ConflictScope::contains`
  (replaces the free fn `file_in_replaced_partition`);
- new `ReplacePartitionsAction::data_spec_id`, `::conflict_scope`, `::is_full_table_replace`;
- `TransactionAction::validate` computes the scope once and threads `&ConflictScope` through
  `validate_added_data_files` / `validate_deleted_data_files` / `validate_no_new_delete_files`;
- `ReplacePartitionsOperation` gains `full_table_replace`; `delete_files` routes through
  `SnapshotProducer::resolve_filter_deletes(&Predicate::AlwaysTrue, true)` on that branch;
- module-doc corrections: the false "static `replaceByRowFilter` … deferred" block replaced with the
  bytecode facts, and the unpartitioned bullet corrected from `isUnpartitioned()` to
  `fields().isEmpty()`.

Docs: the R104 cell in `docs/parity/GAP_MATRIX.md`, and the `replace_partitions.rs` row of
`crates/iceberg/src/transaction/map.md`.

## 3. Tests added (8)

All in the `tests` module of `replace_partitions.rs`. The module previously had **no** test that built a
second partition spec (verified by grep: every `DataFileBuilder` in it hardcoded `.partition_spec_id(0)`),
so the entire spec axis of the `(spec_id, partition)` key was unexercised.

Fixtures (each asserts its own preconditions):

- `make_two_spec_table_of` — spec 0 UNPARTITIONED (`fields()` empty) holding `test/a.parquet`; spec 1
  `identity(x)` holding `test/b.parquet` at `x = 0`.
- `make_all_void_spec_table` — a **V1** table, spec 0 `identity(x)` holding `test/a.parquet`, evolved to
  spec 1 `[void(x)]`. V1 is load-bearing: `UpdatePartitionSpec` substitutes a `Void` field only on V1; on
  V2/V3 the field is removed outright, the field list empties, and the spec is deduplicated back onto an
  existing id, so no distinct all-void spec exists to test. (Discovered empirically — the first draft
  built it on V3 and the precondition assert fired.)

| Test | Pins |
|---|---|
| `..._field_empty_data_spec_replaces_files_of_every_spec` | apply-side full replace crosses specs |
| `..._partitioned_data_spec_keeps_other_specs_files` | apply-side branch does NOT over-fire |
| `..._all_void_data_spec_does_not_full_replace` | apply uses `fields().isEmpty()`, not `isUnpartitioned()` |
| `..._field_empty_data_spec_conflicts_with_other_spec_append` | `alwaysTrue` scope, `validateAddedDataFiles` |
| `..._field_empty_data_spec_conflicts_with_other_spec_delete_file` | `alwaysTrue` scope, `validateNoNewDeleteFiles` |
| `..._field_empty_data_spec_conflicts_with_other_spec_data_removal` | `alwaysTrue` scope, `validateDeletedDataFiles` |
| `..._partitioned_data_spec_keeps_narrow_conflict_scope` | scope does NOT over-fire |
| `..._all_void_data_spec_uses_always_true_conflict_scope` | validate uses `isUnpartitioned()`, not `fields().isEmpty()` |

## 4. Mutations — all nine APPLIED and observed

Harness: `cp` backup (never `git checkout --`, per the 2026-08-08 lesson), apply the mutation, run the
**full** `cargo test -p iceberg --lib`, restore with plain `cp` + `touch` (never `cp -p`), `cmp`-verify.
Every run printed exactly one `Compiling iceberg v0.9.1` line, so no result came from a stale artifact.
A mutation whose target string was absent would have exited 91 (HARD-FAIL, never SKIP); none did.

Green baseline on the restored tree: **3396 passed; 0 failed; 1 ignored** (rebuild confirmed).

| # | Mutation | Result | RED set (from the run's `failures:` block) |
|---|---|---|---|
| M1 | `is_full_table_replace` ⇒ always `false` (delete the apply branch) | 3395/1 | `..._field_empty_data_spec_replaces_files_of_every_spec` |
| M2 | `is_full_table_replace` ⇒ always `true` (over-broaden) | 3382/14 | `..._partitioned_data_spec_keeps_other_specs_files`, `..._all_void_data_spec_does_not_full_replace`, `..._partitioned_data_spec_keeps_narrow_conflict_scope`, + 11 pre-existing single-spec tests (`..._replaces_only_the_added_partition_keeps_others`, `..._replace_empty_partition_is_pure_add`, the cherry-pick replay, …) |
| M3 | apply predicate harmonized onto `is_unpartitioned()` | 3395/1 | `..._all_void_data_spec_does_not_full_replace` |
| M4 | `conflict_scope` ⇒ always `Partitions` (delete the wide branch) | 3392/4 | the three `field_empty_..._conflicts_with_other_spec_*` tests + `..._all_void_data_spec_uses_always_true_conflict_scope` |
| M5 | `conflict_scope` ⇒ always `AllFiles` (over-broaden) | 3392/4 | `..._partitioned_data_spec_keeps_narrow_conflict_scope` + the three pre-existing `..._allows_concurrent_*_in_other_partition` tests |
| M6 | validate predicate harmonized onto `fields().is_empty()` | 3395/1 | `..._all_void_data_spec_uses_always_true_conflict_scope` |
| M7a | `scope.contains` ⇒ `false` at the `validate_added_data_files` site ONLY | 3392/4 | `..._field_empty_data_spec_conflicts_with_other_spec_append`, `..._all_void_data_spec_uses_always_true_conflict_scope`, `..._rejects_concurrent_append_to_replaced_partition`, `..._rejects_concurrent_append_using_tx_captured_starting_snapshot` |
| M7b | same, at the `validate_deleted_data_files` site ONLY | 3394/2 | `..._field_empty_data_spec_conflicts_with_other_spec_data_removal`, `..._rejects_concurrent_deleted_data_in_replaced_partition` |
| M7c | same, at the `validate_no_new_delete_files` site ONLY | 3393/3 | `..._field_empty_data_spec_conflicts_with_other_spec_delete_file`, `..._rejects_concurrent_added_delete_in_replaced_partition`, `..._rejects_concurrent_added_delete_using_tx_captured_start` |

M7a/b/c exist because the three call sites share the byte-identical line
`.find(|file| scope.contains(file))`; the union mutation (M4) reds all four scope tests at once and would
have read as coverage while leaving a site unpinned (the 2026-07-25 "mutate a shared seam at ALL its call
sites" lesson). Each per-site mutation reds the test named for that site and no other site's test — the
anchor-miss tell from the same lesson did not occur. They were anchored on the preceding distinguishing
line (`let added = …` / `let deleted = …` / `let added_deletes = …`), not on the mutated line.

### Non-vacuity accounting — read this as a bound, not as an enumeration

Claimed precisely: **M3 and M6 are the discriminating pair for the asymmetry.** M3 reds exactly one test
(the apply half) and M6 reds exactly one test (the validate half), and they are different tests — so
neither half's assertion is dominated by the other's, and neither predicate can be replaced by the other
without a RED. Likewise M1/M2 and M4/M5 are opposite-direction pairs, each with a non-empty RED set, so
neither branch is a one-way guard.

**NOT claimed:** that the assertion-domination audit inside each individual test is exhaustive. Several
tests assert a live-file SET and an error kind/message in the same body, and no mutation was constructed
to isolate, say, the `!err.retryable()` assertion from the `err.kind()` assertion above it. Those
secondary assertions follow the module's existing convention and are unproven by mutation here.

M2's 14-test RED set is reported as observed; it is not offered as evidence of anything beyond "the
over-broadening direction is loudly covered". Note in particular that
`..._partitioned_data_spec_keeps_narrow_conflict_scope` also asserts a live-file set, so it reds under an
apply-side mutation too — it is not a scope-only pin.

## 5. Residue found and NOT fixed

1. **Multi-spec conflict INTEROP does not exist.** Surveyed the whole harness: every leg that drives
   `ReplacePartitions` (`run-interop-replace-partitions-conflict.sh` / C4,
   `run-interop-write-data.sh` fixture E, `run-interop-write-actions.sh` s4,
   `run-interop-validate-append-only.sh`, `run-interop-s5-isolation.sh`) builds exactly one partition
   spec, and every leg that evolves a spec (`run-interop-multi-spec.sh`,
   `run-interop-multispec-merge.sh`, the `FileScopedDeleteOracle`, `run.sh`) drives only
   `fast_append` / `row_delta`. The two sets are disjoint. Closing it needs an `InteropOracle` mode, a
   `run-interop-*.sh` driver, a Rust `interop_*.rs` consumer and a `SUITE_FLOOR_DEFAULT` 53 → 54 ratchet.
   **This is what keeps R104 at 🟡.**
2. **`dataSpec()`'s two `Preconditions.checkState` throws are not ported.** An empty
   `replace_partitions` and one whose added files span different specs both commit here and both throw
   `IllegalStateException` in Java. `data_spec_id()` returns `None` for both, which routes to the
   NARROWER branch, so the fork can only under-fire relative to Java, never over-fire. Porting the empty
   case would flip the existing pinned test
   `test_replace_partitions_with_no_added_files_adds_nothing`, so it needs its own decision.
3. **`ReplacePartitions` has no `caseSensitive(boolean)` in the Java API**, so the `alwaysTrue` row
   filter is bound with `case_sensitive = true` (the `MergingSnapshotProducer` constructor default).
   `alwaysTrue` binds no column names, so the flag is inert here — stated for completeness, not a gap.
4. **The `apply` branch is reached through `resolve_filter_deletes`, not through a literal port of
   `ManifestFilterManager.filterManifest`.** The two agree on `alwaysTrue` **for the DATA half only** (the
   residual is `alwaysTrue`, strict metrics trivially satisfy it, so every live data file resolves and the
   PARTIAL-match error is unreachable) — and even that is a REASONED equivalence over the shared helper,
   not a separately measured one.
5. **The DELETE half of `deleteByRowFilter` is NOT ported** (found by the independent Critic; residue 4
   originally overstated the equivalence as total). `MergingSnapshotProducer.deleteByRowFilter` drives BOTH
   managers — `filterManager` at offset 10 and `deleteFilterManager` at offset 18 — and
   `MergingSnapshotProducer.apply` runs `deleteFilterManager.removeDanglingDeletesFor(filterManager
   .filesToBeDeleted())` at offsets 103-114 and `deleteFilterManager.filterManifests(...)` at offset 152.
   `ManifestFilterManager`'s `isDeleteManifestReader()` branch permits a PARTIAL match rather than
   erroring, so under `alwaysTrue` Java removes EVERY live delete file. This port's
   `resolve_filter_deletes` skips non-DATA manifests and `existing_manifest` carries every DELETE manifest
   forward unchanged, so after a full-table replace the fork keeps a delete-file set Java deletes. NOT a
   row-resurrection hazard (every referenced data file is removed; replacements carry a higher sequence
   number) and the same gap pre-exists on the `dropPartition` path, but it IS a manifest-list divergence
   byte-level interop would surface, and it is a reason R104 stays 🟡.
6. **New error surface on the full-replace branch** (Critic finding, S4). `resolve_filter_deletes` →
   `build_residual_evaluator` errors with `unknown partition spec id {spec_id}` when a live data file
   carries a spec id absent from table metadata; the previous `resolve_partition_deletes` path silently
   skipped such a file. On malformed or foreign metadata a full replace now hard-errors where it once
   committed. Judged the better posture (fail-closed) but recorded as a behaviour change, not a silent one.

## 6. Reasoned rather than observed

- The Spark dispatch (§1b) is read from bytecode, not from a running Spark job. No Spark job was run.
- Java's *behavior* under the two `dataSpec()` branches is inferred from the bytecode plus the existing
  ported `ManifestFilterManager` semantics. **No Java program was executed in this unit** — that is
  exactly the interop leg listed as residue item 1, and it is why R104 does not flip.
- The pre-change lib-test baseline is derived (3396 observed − 8 added = 3388), not separately measured
  on a stashed tree.
