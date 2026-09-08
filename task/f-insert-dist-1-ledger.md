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

# F-INSERT-DIST-1 — effective partition distribution for DataFusion append

**Date:** 2026-09-07. **Branch:** `codex/fork-insert-dist-1`.
**Base:** `origin/main` `85db42f285703682629b3b53bd1ebcd3091c6bc5`.
**State:** `FINALIZED_PENDING_CI_INTEGRATION`; Docker integration is blocked by local environment
accessibility.
**Path:** STANDARD because this changes a data-write path.

This ledger retires when the fork change merges or the owner removes the unit.

## First checkpoint

The main checkout is clean and remains untouched. The lane is
`/tmp/codex-fork-insert-dist`. The source and `/tmp` share `/dev/nvme1n1p1`,
which had 632 GB free before the worktree was created.

No local branch name matches this insert-distribution unit. The owned fork has
no open pull request. A search of the upstream open pull requests found no
partition-distribution repair.

The defect is in the physical-plan contract. `insert_into` adds a
`RepartitionExec` below `IcebergWriteExec`. DataFusion 54.1 removes manually
inserted distribution-changing nodes during `EnforceDistribution`. It restores
them only from the consuming node's declared requirements. `IcebergWriteExec`
declares an unspecified distribution and returns `false` from
`benefits_from_input_partitioning`, so the optimizer removes the planned hash
exchange and does not restore it. Each source task then opens its own writer for
each partition value.

The existing helper also uses round-robin distribution for temporal-only
partition specs. That does not co-locate equal transformed partition values.
This unit treats every projected `_partition` struct as a valid hash key. The
struct already contains the evaluated Iceberg transform values.

## Implemented minimal fix

`IcebergWriteExec` preserves its semantic distribution and ordering as explicit
input requirements. Its `with_new_children` implementation carries those
requirements while DataFusion rewrites the child tree. The hash requirement
makes DataFusion restore the exchange after optimization. The ordering
requirement keeps the partition sort above that exchange when fanout is off.
Unpartitioned input has no writer distribution or ordering requirement.

`determine_partitioning_strategy` hashes every partitioned table on the
projected `_partition` value. This closes the temporal-transform split without
hashing source values or recomputing transforms inside the writer.

The existing `write.distribution-mode=none` test is not a partitioned-mode test
and the property is not read by production code. Implementing the full Iceberg
distribution-mode property is outside this defect repair unless the
orchestrator expands the unit. The regression suite will preserve the current
unpartitioned round-robin behavior and both fanout-order modes.

## File allowlist

- `crates/integrations/datafusion/src/physical_plan/write.rs`
- `crates/integrations/datafusion/src/physical_plan/repartition.rs`
- `crates/integrations/datafusion/src/physical_plan/map.md`
- `crates/integrations/datafusion/tests/insert_distribution.rs`
- `crates/integrations/datafusion/tests/map.md`
- `task/f-insert-dist-1-ledger.md`

`Cargo.toml`, `Cargo.lock`, every dependency file, `.github/`, catalog code,
and RePark stay closed. A new required path stops the unit for re-scope.

## Proposition ledger

The first checkpoint proved that each clause was feasible and checkable. In the
table below, `EXECUTION PROVEN` means the implemented patch has direct passing
evidence. `PENDING FINAL DIFF` is not an implementation attestation.

| Clause | Checkable proposition | Proof obligation | Status |
|---|---|---|---|
| C-001 | The optimized public `INSERT INTO ... SELECT` plan retains a hash exchange on the projected partition value at `target_partitions = 8`. | Inspect the optimized physical plan through DataFusion. The pin is red on the base because the exchange is absent. | EXECUTION PROVEN |
| C-002 | Equal transformed partition values from four source tasks reach one writer task for identity, bucket, day, and null partition values. | Public identity/null and combined bucket/day SQL fixtures. Each source task carries every value. Read manifest tuples and record counts. | EXECUTION PROVEN |
| C-003 | The controlled eight-value fixture writes at most eight live data files when the target file size exceeds the fixture. | Base and fixed file census. Assert one file per present value only for this bounded fixture. Do not claim a universal file-count guarantee. | EXECUTION PROVEN |
| C-004 | The append preserves every row and Arrow-visible type with no loss or duplication. | Read through the public table provider. Compare sorted rows, nulls, exact count, and a stable integer sum. | EXECUTION PROVEN |
| C-005 | Fanout enabled keeps the no-sort plan. Fanout disabled hashes first and sorts each writer's input by partition value. | Optimized-plan assertions plus successful clustered-writer execution over interleaved source tasks. | EXECUTION PROVEN |
| C-006 | Unpartitioned append and `target_partitions = 1` retain their current behavior. | Public SQL controls and physical-plan shape assertions. | EXECUTION PROVEN |
| C-007 | A child-stream error reaches the caller and no snapshot commits. Distribution adds no unbounded unit-owned buffer. | A failing execution-plan input followed by catalog snapshot inspection; source inspection of DataFusion's backpressured, memory-pooled, spill-capable repartition channel. | EXECUTION PROVEN |
| C-008 | No dependency, workflow, public Rust API, on-disk format, or RePark pin changes. | Final diff and gate inspection. | EXECUTION PROVEN |
| C-009 | Empty public INSERT preserves Iceberg append semantics, and a partitioned helper input without `_partition` fails loudly. | SQL `WHERE false` result, empty snapshot and file census, plus the repartition Plan-error boundary. | EXECUTION PROVEN |

The quantified domain for C-002 is the finite set `{identity, bucket, day,
null identity}` across `{four source tasks, multiple batches per task}`. C-005
enumerates `{fanout enabled, fanout disabled}`. C-006 enumerates
`{unpartitioned, one target partition}`.

## Plan

- [x] Establish the isolated worktree and check disk and overlap.
- [x] Locate the optimizer contract that removes the planned exchange.
- [x] Add public-execution regression pins and record the base failures.
- [x] Implement the declared distribution and ordering contract.
- [x] Run targeted green tests and mutations.
- [x] Run the fork unit gate.
- [x] Run the full allowed offline gates and record all omitted service gates.

## Base-red evidence

`CARGO_TARGET_DIR=/tmp/codex-fork-insert-dist-target CARGO_BUILD_JOBS=2 cargo test
-p iceberg-datafusion --test insert_distribution --locked -- --nocapture` exited 101 on
the exact base. The public insert reported 32 rows. The read-back returned 32 rows,
28 non-null partition values, and `sum(id) = 496`. The manifest census contained
eight values with four one-row files each, for 32 files total. The controlled
expectation is one four-row file for each value. This is one red test out of one.

`CARGO_TARGET_DIR=/tmp/codex-fork-insert-dist-target CARGO_BUILD_JOBS=2 cargo test
-p iceberg-datafusion --lib
physical_plan::repartition::tests::test_temporal_partition_values_use_hash_partitioning
--locked -- --exact --nocapture` exited 101 on the exact base. The Day-only
partition strategy was `RoundRobinBatch(4)`. This is one red test out of one and
is independent of the public identity/null fixture.

## Focused execution evidence

The public regression file contains seven tests. Two partitioned fixtures each use
four source tasks and two batches per task. The identity/null fixture reads all
32 rows in `id` order and compares the exact `Int32`, nullable `Int32`, and UTF-8
values. Its manifest has eight live files, one four-row file for each fixture
partition value. The combined bucket/day fixture uses IDs 0 and 3, which evaluate
to distinct values under `bucket[4]`, across two Day values. Its four evaluated
partition tuples must each have one four-row file.

The fanout-enabled optimized plan retains a hash exchange without a writer-input
sort. The fanout-disabled optimized plan has a sort directly under the writer
and retains a hash exchange. Both writer requirements survive a schema-preserving
`with_new_children` replacement. A one-target partition run
keeps all rows and eight fixture-bound files. An unpartitioned run keeps the
writer requirement unspecified and returns all eight rows with the expected
sum. The zero-row control requires a one-row `UInt64(0)` result, an empty current
snapshot, no live data files, and a zero table count. A controlled child execution
error reaches the caller and the table has no current snapshot afterward.

DataFusion 54.1's repartition implementation uses backpressure channels. Each
output partition registers a memory consumer that can spill, and sends wait
when every receiver channel already has data. The fork change adds no queue,
buffer, or row materialization of its own.

## Mutation evidence

- Replacing the writer hash requirement with `UnspecifiedDistribution` made
  four of six public regression tests red. The two controls stayed green.
- Replacing the clustered writer ordering requirement with `None` made its
  focused public test red, zero of one passing.
- Restoring temporal-only round-robin made the independent Day strategy pin
  red, zero of one passing.
- Replacing the writer hash requirement with `UnspecifiedDistribution` made
  the distinct bucket/day public test red before execution, zero of one passing.
- Changing the missing-partition Plan error text made the direct boundary pin
  red, zero of one passing.

Every mutation was temporary and was reversed before validation. The final
focused rerun recompiled both restored source files and passed.

## Critic resolution

The 2026-09-08 review transcript is
`/tmp/codex-glm-critics-20260907/iceberg-insert-critic-transcript.md`.

| Finding | Disposition | Evidence or action |
|---|---|---|
| C1: stored column indices can become stale after schema-changing `with_new_children` | REFUTED | DataFusion physical parents such as `FilterExec`, `ProjectionExec`, and `SortExec` retain positional expressions across child replacement. Reachable optimizer replacements preserve the child schema slots used by those expressions. The public append pipeline creates `_partition` before the writer. The test now replaces the child with an empty plan that has the same schema; arbitrary schema-changing children are outside the `ExecutionPlan` replacement contract. |
| C2: zero-row INSERT was unpinned | CONFIRMED, RESOLVED | Added a public SQL control for the count result, empty snapshot, empty live-file census, table row count, and Arrow-visible result types. |
| C3: missing or malformed `_partition` was unpinned | PARTLY CONFIRMED, RESOLVED | The partitioned repartition helper's missing-column Plan error now has a direct pin. A malformed projected type is not reachable through public INSERT because `project_with_partition` constructs and validates the projected struct. No private-helper mirror was added for an unreachable shape. |
| C4: IDs 0 and 1 collapse to one `bucket[4]` value | CONFIRMED, RESOLVED | The fixture now uses IDs 0 and 3, producing two bucket values across two Day values and four manifest tuples. A nested source field cannot use the quantified identity, bucket, or day transforms; the builder rejects nonprimitive inputs. Fanout-disabled at one target is an unnecessary cross-product of independently covered controls. |
| C5: distribution-mode gap lived only in this ledger | CONFIRMED, RESOLVED | `physical_plan/map.md` now records that `none`, `hash`, and `range` are not interpreted. No property semantics or parity status changed. |
| Duplicate fanout parsing | OUT OF SCOPE | The public table path validates the property before this writer lookup. Parsing cleanup and property semantics remain separate work. |
| `sort.rs` null-order prose | OUT OF SCOPE | The statement predates this patch, is outside the allowed repair, and does not change append distribution behavior. |
| Base-red “one out of one” arithmetic | REFUTED | The base-red command ran when the integration target contained one test. The later six-test matrix did not exist at that checkpoint. |
| Direct hash adjacency under the writer | WORDING CORRECTED | The test pins a retained hash exchange and the presence or absence of the required writer-input sort. It does not claim direct hash adjacency in fanout mode. |

## Validation

All cargo commands used `CARGO_TARGET_DIR=/tmp/codex-fork-insert-dist-target`
and `CARGO_BUILD_JOBS=2`. The 2026-09-08 critic-resolution runs also held
`flock /tmp/codex-glm-critics-20260907/build.lock`.

| Command | Result |
|---|---|
| `cargo test -p iceberg-datafusion --test insert_distribution --locked -- --nocapture` | exit 0; 7 passed |
| `cargo test -p iceberg-datafusion --lib physical_plan::repartition::tests --locked -- --nocapture` | exit 0; 13 passed |
| `make check` | exit 0; format, workspace all-target/all-feature clippy with denied warnings, TOML, machete, artifact, matrix, comment-block, and Rust-size gates passed |
| `make unit-test` | exit 0; workspace all-feature doc tests and library tests passed |
| `git diff --check` | exit 0 |

`make test` did not run during this validation stage. After finalization lifted
the earlier agent fence, the canonical attempt exited 2 in `docker-up` before
test execution. The Docker integration gate remained blocked because
the configured desktop socket was absent, access to the system daemon was
denied, and port 9000 was occupied. The Java interoperability suites were not
run because they require their external fixtures and runtimes. No AWS or
credentialed gate ran. The changed integration test executes locally against
`MemoryCatalog` and local temporary storage.

The final allowlist inspection contains only the six declared paths. It changes
no manifest, lockfile, workflow, public API, or serialized Iceberg format.

## Distribution-mode boundary

The fork does not currently interpret `write.distribution-mode`. The old test
name implied `none` support, but its table was unpartitioned and production did
not read the property. The test now states only the observed unpartitioned
round-robin behavior. This unit does not add `none`, `hash`, or `range` property
semantics and makes no parity claim for them.

## Resource record

The filesystem had 632 GB free before worktree creation, 628 GB before the first
build, 693 GB before initial focused validation, 690 GB before the initial broad
unit gate, 650 GB before critic-resolution builds, and 645 GB before the final
broad unit gate. The task-owned target directory remains for the independent
critic. No other task's worktree, cache, or uncommitted file was removed. The
main checkout remains clean and was not edited.

## Pre-execution review

```yaml
SELF_LOGIC_REVIEW:
  id: SLR-F-INSERT-DIST-1-SCOPE
  agent: Actor
  action: Define the append distribution repair and its regression surface
  charter_trace: C-001 through C-008
  preconditions:
    - Exact fork base is checked out in an isolated lane: SATISFIED
    - No owned-fork or upstream pull request overlaps the unit: SATISFIED
    - DataFusion removes manual repartition nodes without a consumer requirement: SATISFIED
    - Production edits remain blocked until orchestrator release: SATISFIED
  expected_output: A reviewable scope and a base-red test plan
  success_condition: The orchestrator can adopt or narrow the allowlist before production edits
  step_risks:
    - Hashing source values instead of evaluated transform values: HANDLED by hashing `_partition`
    - Repartition after sort breaks clustered writer order: HANDLED by declaring both requirements
    - Universal one-file claim becomes flaky: HANDLED by a bounded fixture and threshold
    - A write error commits partial metadata: HANDLED by a no-snapshot error pin
  contingencies:
    - Scope rejection: EXECUTABLE by an additive ledger amendment before code changes
  tripwire_scan: CLEAN
  uncertainty: NONE
  verdict: PROCEED
  escalation: "—"
```

## Final independent review and draft-PR readiness, 2026-09-08

The candidate source, test, and map bytes did not change between the Actor gates and the final
independent Grok review. Grok issued `CLEAN` at S2 with complete charter coverage after compiling
and running the public insert-distribution tests (7 passed), repartition unit tests (13 passed),
the focused writer test (1 passed), and adjacent partitioned INSERT SELECT tests (12 passed). This
closeout changes ledger prose only.

The review independently confirmed evaluated `_partition` hashing, temporal and bucket/day
strategies, writer distribution and ordering requirements, schema-preserving child replacement,
missing-column failure, zero-row behavior, row and manifest-file correctness, and controlled error
rollback. It withdrew the schema-changing-child concern because the reachable DataFusion physical
parents retain positional expressions across schema-preserving replacement. It confirmed the
remediations for zero-row coverage, the missing `_partition` error pin, distinct bucket fixtures,
and the durable distribution-mode limitation.

The accepted S3 advisories remain: direct writer construction falls back to unspecified
distribution when `_partition` is absent; fanout is parsed twice after public validation; existing
sort prose differs from `SortOptions::default` while both execution paths use the same options;
and full `write.distribution-mode` semantics remain unsupported and out of scope. The Actor
mutations made the writer hash requirement, clustered ordering, Day strategy, bucket/day strategy,
and missing-column message pins red, then the restored focused populations passed.

The Actor's `make check` and `make unit-test` runs exited 0. Final readiness checks also exited 0
for `typos .`, `make check-msrv`, the no-default-features `iceberg` build, and
`cargo deny check advisories`; the advisory run retained the existing nonfatal yanked-crate warning
for `chacha20` 0.10.1. The frozen base and fetched `origin/main` were both
`85db42f285703682629b3b53bd1ebcd3091c6bc5`.

At this pre-commit checkpoint, no source delta had followed independent review and no commit, push,
pull request, or merge had occurred. The canonical `make test` attempt exited 2 in `docker-up`
before test execution and did not alter Docker resources. The Docker integration gate remains
blocked by local environment accessibility. No Docker pass, R7 waiver, full R7 readiness, merge,
or delivery is claimed.

## Finalization (Grok 4.6, 2026-09-08)

Fresh clone `/tmp/grok-insdist` on `codex/fork-insert-dist-1` at
`aae7bd2cbb096dc33c3cee3924f53796e6b29551`. `git fetch origin main` left `origin/main` at
`85db42f285703682629b3b53bd1ebcd3091c6bc5`. `git merge-base --is-ancestor origin/main HEAD`
exited 0. `git log --oneline origin/main..HEAD` showed exactly one commit before this
finalization commit. Registry reachable; no `--offline`.

A fresh-eyes re-read of `origin/main...HEAD` found no merge-blocking comment in code, no
literal home path, no dependency or `.github/` change, and no test that cannot fail.

| Command | Result |
|---|---|
| `CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=8 make check` | exit 0; fmt, workspace all-target/all-feature clippy with denied warnings, TOML, machete, artifact, matrix, comment-block, and Rust-size gates passed. Python file-size unittest: `Ran 11 tests in 0.012s` / `OK`. No cargo `test result:` line. |
| `CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=8 cargo test -p iceberg-datafusion --test insert_distribution --locked` | exit 0; `test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.17s` |
| `CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=8 cargo test -p iceberg-datafusion --lib physical_plan::repartition --locked` | exit 0; `test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 204 filtered out; finished in 0.01s` |
| `CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=8 cargo test -p iceberg-datafusion --lib physical_plan::write --locked` | exit 0; `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 216 filtered out; finished in 0.01s` |
| `CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=8 cargo test -p iceberg-datafusion --test partitioned_insert_select_test --locked` | exit 0; `test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s` |
| `typos .` | exit 0 |

The integration gate is the PR's CI `Tests (default)` run after F-HMS merges. The
orchestrator records the run id and head sha in the PR before merging. The red `Tests
(default)` leg on this draft is outside the branch: `dev/hms/Dockerfile` dies in
`apt-get update` because Debian 11 (`bullseye-security`) left long-term support on
2026-08-31 and its release file expired on 2026-09-07. F-HMS on `fix/hms-fixture-download`
replaces that apt step with a checksum-pinned download stage.
