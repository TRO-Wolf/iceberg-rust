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

# Evidence ledger — PR-1 REPLACE record-count invariant (C-002 / row R107)

Capability status lives on GAP_MATRIX row R107. This file holds the Java decode,
unit pins, mutation counts, interop command, and the section 9 delivery template.

Plan of record: `task/iceberg-v3-production-work-plan-2026-09-01.md`.

## Charter

```yaml
LEDGER:
  - id: C-002
    proposition: >
      Every replace snapshot rejects added-records > deleted-records. Missing
      summary keys count as zero.
    verdict: PROVEN
    evidence: >
      Shared SnapshotProducer::commit calls
      replace_record_count::validate_replace_record_counts after summary and
      before ManifestListWriter / manifest_file IO. Pins below.
  - id: C-007
    proposition: >
      Every claimed test proves its cited behavior. Negative guards have a
      mutation that turns the test red.
    verdict: PROVEN
    evidence: four one-knob mutations, each 1 red out of 1, restored after.
```

## Java decode (iceberg-core 1.10.0)

Jar: `~/.m2/repository/org/apache/iceberg/iceberg-core/1.10.0/iceberg-core-1.10.0.jar`

Class: `org.apache.iceberg.SnapshotProducer`

Method: `public Snapshot apply()`

Constant pool:

- `#333` / `#334` = `"replace"`
- `#342` / `#343` = `"added-records"`
- `#349` / `#350` = `"deleted-records"`
- `#351` / `#352` = `"Invalid REPLACE operation: %s added records > %s replaced records"`
- `#344` = `PropertyUtil.propertyAsLong(Map, String, long)`

Bytecode (offsets 294-367):

```
294: invokevirtual summary:()Ljava/util/Map;
300: invokevirtual operation:()Ljava/lang/String;
311: ldc_w "replace"
316: invokevirtual String.equals
319: ifeq 367
324: ldc_w "added-records"
327: lconst_0
328: invokestatic PropertyUtil.propertyAsLong
335: ldc_w "deleted-records"
338: lconst_0
339: invokestatic PropertyUtil.propertyAsLong
348: lcmp
349: ifgt 356          // fail when added > deleted; equal is valid
357: ldc_w "Invalid REPLACE operation: %s added records > %s replaced records"
364: invokestatic Preconditions.checkArgument:(ZLjava/lang/String;JJ)V
```

`PropertyUtil.propertyAsLong(Map, String, long)` (offsets 0-24): `map.get(key)`; null → default `0`; else `Long.parseLong`.

`BaseRewriteFiles.operation()` and `BaseRewriteManifests.operation()` both `ldc "replace"; areturn`.

Java writes manifests first (`apply(TableMetadata, Snapshot)` at offset 59, `manifestListPath` at 64, close at 189) and only then runs the record-count check (offsets 311-364). An invalid 3-to-5 replacement therefore leaves orphan manifests on the Java side; the snapshot pointer does not move. The fork is outcome-equal (both refuse, neither moves the pointer) and stricter on placement: `SnapshotProducer::commit` completes summary, runs the guard, and only then constructs `ManifestListWriter` / calls `manifest_file()`, so a refused REPLACE writes no metadata object. That placement is the plan PR-1 requirement, not a claim of byte-level equality on the refused path.

## Unit pins

| Test | Assertion |
|---|---|
| `replace_rejects_added_records_greater_than_deleted_records` | 3-row file replaced by 5-row file is `DataInvalid`; message matches Java; snapshot and metadata pointer unchanged; avro count under metadata/ unchanged |
| `replace_commits_when_added_records_equal_deleted_records` | 3 → 3 commits as `Operation::Replace` |
| `replace_commits_when_added_records_trail_deleted_records` | 5 → 3 commits as `Operation::Replace` |
| `rewrite_manifests_replace_commits_when_record_count_keys_are_absent` | `RewriteManifests` cluster-by commits; `added-records` and `deleted-records` absent |
| `replace_still_refuses_on_retried_attempt_after_conflict` | first attempt is a valid 5-to-5 replace; catalog CAS fails after a concurrent shrink of the original to 3 rows; the retried attempt is `DataInvalid` (`5 > 3`); avro count equals the post-conflict count |

Command: `cargo test -p iceberg --locked --lib replace_`

Result: the five pins passed (filter also hits unrelated `replace_` names; the five named tests are green).

## Mutations (one knob, restore after)

Each HARD-FAIL if the pattern is absent or the mutant has no assertion signal. Restored from `.bak`, `touch`ed, md5-identical.

| # | Knob | Command | Result |
|---|---|---|---|
| 1 | `if added > deleted` → `if false` in `replace_record_count.rs` | `cargo test -p iceberg --locked --lib replace_rejects_added_records_greater_than_deleted_records` | **1 red out of 1**. Panicked at `expect_err("replacing 3 rows with 5 rows must be DataInvalid")` |
| 2 | move `validate_replace_record_counts` to after `manifest_file()` in `snapshot.rs` | same command | **1 red out of 1**. `refused REPLACE must not write a new manifest or manifest-list object` left 4 right 2 |
| 3 | `None => Ok(0)` → `None => Err(...)` in `property_as_long` | `cargo test -p iceberg --locked --lib rewrite_manifests_replace_commits_when_record_count_keys_are_absent` | **1 red out of 1**. `RewriteManifests REPLACE with absent record-count keys must commit: DataInvalid => missing snapshot summary property added-records` |
| 4 | skip the comparison after the first `Operation::Replace` call (`FIRST_REPLACE_CHECK` swap) | `cargo test -p iceberg --locked --lib replace_still_refuses_on_retried_attempt_after_conflict` | **1 red out of 1**. Panicked at `expect_err("retried invalid REPLACE after a conflict refresh must still be DataInvalid")` — the retried 5-to-3 attempt committed |

## Interop

Command: `dev/java-interop/run-interop-replace-invariant.sh`

Result: exit 0. Java generate + Rust GEN + Java verify `0 failures` + Rust D1 + fixture count **3/3**.

Expected fixture count: **3**

- `invalid/threw.json` — Java 3-to-5 throws
- `valid_java/java_rows.json` — Java 3-to-3 rewrite rows
- `valid_rust/rust_table/metadata/final.metadata.json` — Rust 3-to-3 rewrite

Java mode: `generate-interop-replace-invariant` / `verify-interop-replace-invariant`.

Maven: `/opt/maven/bin/mvn -o`, `JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`.

Docker `make test` legs: unavailable, excused.

## Gates

| Command | Exit |
|---|---|
| `make check` | 0 |
| `cargo test -p iceberg --locked` | 0 |
| `make check-matrix-anchors` | 0 |
| `dev/java-interop/run-interop-replace-invariant.sh` | 0 |

Docker `make test` legs: unavailable, excused.

## CI-only evidence gap

None for this unit. Docker `make test` legs are excused (no Docker). Glue / S3 Tables credentialed runs are out of scope.

## Breaking public API change

None. The guard is `pub(crate)` on the existing `SnapshotProducer::commit` path.

## Section 9 delivery template

```text
Charter clauses: C-002, C-007
Matrix rows: row R107
Java methods or bytecode read: SnapshotProducer.apply() offsets 311-364; PropertyUtil.propertyAsLong(Map,String,long); BaseRewriteFiles.operation(); BaseRewriteManifests.operation()
Files changed: crates/iceberg/src/transaction/{snapshot.rs,replace_record_count.rs,replace_record_count_tests.rs,map.md}; crates/iceberg/tests/{interop_replace_invariant.rs,map.md}; dev/java-interop/{InteropOracle.java,run-interop-replace-invariant.sh,map.md}; docs/parity/GAP_MATRIX.md; scripts/run_interop_suites.sh; task/pr1-replace-invariant-ledger.md; task/todo.md
Behavior before: a RewriteFiles replace with added-records > deleted-records committed
Behavior after: the shared producer refuses that shape with DataInvalid before manifest IO; equal and shrinking replaces still commit; RewriteManifests with absent keys still commits
Negative cases: 3-to-5 rewrite; conflict-then-retry (retried 5-to-3 after shrink); missing-key handling
Test command and population: cargo test -p iceberg --locked --lib replace_ (five named pins green); cargo test -p iceberg --locked
Mutations, one at a time: (1) remove comparison → 1 red out of 1; (2) move guard below manifest_file → 1 red out of 1 (avro 4 vs 2); (3) missing key as error → 1 red out of 1; (4) skip comparison after the first Replace call → 1 red out of 1 (retried 5-to-3 committed)
Java interop command and fixture count: dev/java-interop/run-interop-replace-invariant.sh ; 3 fixtures
CI-only evidence gap: Docker make test legs excused
Breaking public API change: none
Critic attestation: (orchestrator)
Open findings and dispositions: none at Actor hand-back
```
