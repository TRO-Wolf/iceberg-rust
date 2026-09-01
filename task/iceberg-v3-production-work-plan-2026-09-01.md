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

# Iceberg V3 production work plan for the iceberg-rust fork

**Date:** 2026-09-01

**Reviewed fork revision:** `00cdde00685bbc94552b29fcf8ed6767fe051ce6`

**Consumer reviewed:** RePark `75f5ee35f4355f8a9a3d03ccc77cc751a9610f7a`

**State:** APPROVED WITH AMENDMENTS (owner, 2026-09-01, via the RePark orchestrator; the amendments in section 11 are binding and the fork agent applies them to sections 3, 4 and 5 before the first unit starts)

**Scope:** iceberg-rust work only

This plan converts the 2026-09-01 V3 production audit into checkable fork work. It does not own
capability status. Current status stays in `docs/parity/GAP_MATRIX.md`. This plan cites matrix rows
by permanent `R<id>` only.

This plan supersedes the work-scope conclusions in `task/v3-production-readiness-audit.md`. That
file remains the historical record of the earlier audit at revision `d62fe54bd`.

## 1. Release decision

The reviewed fork is not ready for RePark's full V3 production gate. The work is seven units:
three confirmed correctness repairs, one catalog conformance and repair unit, two composed interop
units, and one production closeout. Type and format extensions need an owner decision before they
become fork work.

Do not make a production-ready claim until all mandatory units in section 4 meet their exit gates.
An independent Critic must converge each PR. The final bundle also needs an independent
bundle-scope review.

## 2. Scope contract

### 2.1 Mandatory outcomes

| Clause | Checkable proposition | Necessity evidence |
|---|---|---|
| C-001 | `RewriteDataFiles` writes each output row under the partition tuple computed from the selected output spec. It never copies an old tuple into a different spec. | Current planner and writer can mislabel rows after same-arity spec evolution. Matrix row R135 owns the action. |
| C-002 | Every `replace` snapshot rejects `added-records > deleted-records`. Missing summary keys count as zero. | Java rejects this shape. The fork accepts it. Matrix row R107 owns the action. |
| C-003 | Every V3 DataFusion UPDATE path keeps `_row_id`. An updated row advances `_last_updated_sequence_number`. | The merge-on-read path omits both lineage columns. Matrix row R166 owns row lineage. |
| C-004 | V2 to V3 upgrade and every required V3 maintenance action preserve live rows, delete semantics, row lineage, and snapshot validity across Java and Rust. | Unit coverage does not close the production interop claim. Matrix rows R109, R114, R135, R136, and R166 own the parts. |
| C-005 | Glue and S3 Tables classify conflict, retryable failure, and unknown commit outcome without duplicate commits or unsafe cleanup. | Offline classification exists. Credentialed conformance does not. Matrix rows R110 and R157 own the parts. |
| C-006 | Every required V3 reference operation reads and commits the named branch without moving `main`. Java and Rust agree on the result. | Current branch work has unit coverage but no Java interop. Matrix row R168 owns the capability. |
| C-007 | Every claimed test proves its cited behavior. Negative guards have a mutation that turns the test red. Interop harnesses hard-fail when their mutation or fixture cannot run. | Green suites do not exercise the two silent-corruption cases or merge-on-read lineage. |

### 2.2 Production envelope used by this plan

The mandatory plan assumes this RePark V1.0 envelope:

- Parquet data files.
- One application writer per table.
- AWS Glue and S3 Tables are the priority catalogs.
- V2 tables can upgrade to V3.
- Legacy V2 position-delete files in the upgrade path are Parquet. ORC and Avro position-delete
  conversion needs a separate owner-selected charter.
- V3 uses Puffin deletion vectors.
- V3 supports append, DELETE, UPDATE, MERGE, and the required maintenance actions.
- Partition evolution remains supported.
- Java and Spark remain the interoperability oracle.

Changing this envelope requires a new scope audit. Section 6 records the known decision points.

## 3. Dependency order

```text
PR-6A branch interop (first)

PR-1 replace invariant          PR-3 row-DML lineage        PR-5A harness (offline)
  |                               (MoR UPDATE + F-rp3-c7)      |
  +--> PR-2 partition-safe          |                          |
         RewriteDataFiles           +--> PR-6B MoR UPDATE      |
         |                          |    lineage branch cell   |
         +--------------------------+--> PR-4 upgrade and      |
                                         maintenance interop   |
                                                               +--> PR-5A credentialed run

PR-1..PR-6B, PR-5A --> PR-7 production evidence closeout
PR-5B (throttle matrix, sub-SDK connector) --> deferred to the reliability / multi-writer track
```

Order of record (section 11.3): PR-6A immediately; PR-1, the expanded PR-3 and the PR-5A
harness in parallel; PR-2 after PR-1; PR-4 after PR-2 and PR-3; PR-6B after PR-3; PR-5A
credentialed execution; PR-7. PR-3 has no dependency on PR-1; overlap in snapshot accounting is
merge coordination. PR-1 through PR-6B are STANDARD units. They touch data integrity, public
behavior, or external catalogs. PR-7 is also STANDARD because it closes the production claim.

## 4. Mandatory fork units

### PR-1: Enforce the Java REPLACE record-count invariant

**Owns:** C-002 and the relevant part of C-007

**Matrix:** row R107

**Depends on:** none

#### Implementation

1. Add the invariant at the shared snapshot-producer layer.
2. Apply it only when `operation() == Operation::Replace`.
3. Read `added-records` and `deleted-records` from the completed summary.
4. Treat an absent key as zero. This matches Java `propertyAsLong` behavior.
5. Run the guard immediately after summary completion and before manifest or manifest-list IO.
6. Return `ErrorKind::DataInvalid` when added records exceed deleted records.
7. Keep `RewriteManifests` valid when both keys are absent.
8. Keep valid `RewriteFiles` commits valid when added records equal or trail deleted records.

The guard must not live only in `RewriteFilesAction`. The shared layer covers every producer that
emits `Operation::Replace`. It also covers replay paths that retain that operation.

#### Expected files

- `crates/iceberg/src/transaction/snapshot.rs`
- `crates/iceberg/src/transaction/rewrite_files.rs`
- A transaction test module if the current inline tests exceed the file-size ceiling.
- `dev/java-interop/` and `crates/iceberg/tests/` for the parity leg.
- `docs/parity/GAP_MATRIX.md` row R107 after the evidence exists.
- The touched directories' `map.md` files when their routing text changes.

#### Required tests

| Test | Required assertion |
|---|---|
| Invalid replacement | Replacing a 3-row file with a 5-row file returns `DataInvalid`. The snapshot and metadata pointer do not change. No new manifest or manifest-list object is written. Already staged replacement data files are outside this assertion. |
| Equal replacement | Replacing 3 rows with 3 rows commits. |
| Shrinking replacement | Replacing 5 rows with 3 rows commits. |
| Missing summary keys | A `RewriteManifests` replace with no data counts still commits. |
| Retry | A retry cannot bypass the guard after the base refreshes. |

#### Test adequacy

- Remove the comparison. The invalid-replacement test must turn red.
- Move the guard below manifest creation. The zero-new-metadata-object assertion must turn red.
- Change missing-key handling from zero to error. The `RewriteManifests` test must turn red.
- Record each mutation as `N red out of M` for the exact command used.

#### Interop gate

Run the same invalid replacement through Java and Rust. Both engines must refuse it. Run one valid
replacement in each direction. The other engine must read the resulting table with the same rows.

#### Exit gate

- Unit and mutation evidence are green.
- Java interop is green.
- Matrix row R107 records the evidence.
- Independent Critic converges with no open S0, S1, or S2 finding.

### PR-2: Make RewriteDataFiles partition-safe after spec evolution

**Owns:** C-001 and the relevant parts of C-004 and C-007

**Matrix:** row R135

**Depends on:** PR-1

#### Current failure

`plan_file_groups` sends every non-default-spec task to one empty partition bucket.
`write_compacted_files` then uses the current default spec and the first task's old partition tuple.
It checks tuple arity only. A same-arity source or transform change can therefore stamp every output
file with the wrong current-spec partition value.

#### Implementation

1. Keep candidate selection and bin packing separate from output partition routing.
2. Build a `RecordBatchPartitionSplitter` from the selected output spec and current schema.
3. Read each group's live rows with merge-on-read deletes applied.
4. Preserve stored V3 lineage columns in the read schema.
5. Compute the output partition key from each output row.
6. Add a bounded partition router. Set `max_open_partition_writers` on the action, default it to
   `64`, and reject zero.
7. Evict the least-recently-used writer at the bound. Close it, retain its output files, and open a
   new writer if the same key appears later. Multiple output files for one key are valid.
8. Do not use the current unbounded `FanoutWriter` for this action.
9. Stamp each file with the partition key's spec ID and tuple.
10. Remove `group_partition_tuple` as an output source. Keep it only if another caller proves a
   same-spec contract.
11. Preserve the starting snapshot's data sequence number.
12. Keep file-scoped delete removal and shared-Puffin sibling handling in the same atomic replace.

The existing compute and split pattern in `maintenance/partition_key_audit.rs` is the routing
reference. Its unbounded fanout lifetime is not the production writer contract for this action.
Keep the bounded router private to maintenance unless a second caller needs the same contract.

#### Expected files

- `crates/iceberg/src/maintenance/rewrite_data_files.rs`
- `crates/iceberg/src/maintenance/rewrite_data_files_plan.rs`
- `crates/iceberg/src/maintenance/rewrite_data_files_write.rs`
- A bounded partition-router module under `crates/iceberg/src/maintenance/`.
- A dedicated evolved-spec test module under `crates/iceberg/src/maintenance/`
- `crates/iceberg/src/maintenance/map.md`
- `dev/java-interop/` and `crates/iceberg/tests/`
- `docs/parity/GAP_MATRIX.md` row R135 after the evidence exists.

#### Required test partition

| Evolution class | Required shape | Required proof |
|---|---|---|
| Source field changes, same arity | `identity(x)` to `identity(y)` | At least two old partitions co-enter a rewrite. Full scan and partition-pruned scans return the same correct rows. Output tuples equal recomputed `y`. |
| Transform changes, same source | `identity(x)` to `bucket(x)` | Output tuples equal the current bucket transform. No old identity value is stamped as a bucket value. |
| Transform changes, same arity | `bucket[8](x)` to `truncate[10](x)` | Output tuples equal the current transform for every row. |
| Partitioned to unpartitioned | one-field spec to an empty spec | Output files use the current unpartitioned spec ID and empty tuples. |
| Unpartitioned to partitioned | empty spec to `identity(x)` | Rows fan out to every recomputed current partition. |
| Mixed current and old files | files from both specs in one action | Every output file uses the current spec and a recomputed tuple. |
| Default writer bound | omit `max_open_partition_writers` | The resolved value is exactly `64`. Peak open writers obey it. |
| Invalid writer bound | set `max_open_partition_writers` to zero | Return `DataInvalid` before an output data file is written. |
| High-cardinality output | at least `10 * max_open_partition_writers` current keys in one input group | Peak open writers never exceeds the configured bound. Reopened keys keep all rows and valid tuples. |
| V3 lineage | any evolved-spec rewrite | `_row_id` stays stable. `_last_updated_sequence_number` stays correct for unchanged rows. |
| Deletes | old files with equality deletes, legacy position deletes, or DVs | The rewrite contains only live rows. Superseded file-scoped deletes are removed without dropping siblings. |

#### Test adequacy

- Restore the static `group.first()` tuple. The same-arity source-field test must turn red.
- Skip the partition splitter. The unpartitioned-to-partitioned test must turn red.
- Change the default from `64`. The default-bound assertion must turn red.
- Accept zero. The invalid-bound error and zero-output-file assertions must turn red.
- Remove the eviction path. The high-cardinality peak assertion must turn red.
- Drop accumulated output from an evicted writer. The row census must turn red.
- Drop lineage projection. The V3 lineage test must turn red.
- Disable merge-on-read delete application. Each delete-class test must turn red.

#### Interop gate

1. Java creates data under an old spec. Rust evolves the spec and compacts it. Java must return the
   same live rows for a full scan and for every affected partition predicate.
2. Rust creates the old-spec table. Java evolves the spec and rewrites it. Rust must return the same
   rows and partition metadata.
3. Include one V3 table. Compare `_row_id` before and after Rust compaction.

#### Exit gate

- Every evolution class has a non-vacuous test.
- Both interop directions pass.
- Row identity and delete semantics pass in the evolved-spec fixture.
- Matrix row R135 records the evidence and no longer hides this defect under `output-spec` residue.
- Independent Critic converges with no open S0, S1, or S2 finding.

### PR-3: Preserve row lineage in V3 merge-on-read UPDATE and repair rewrite-aware row allocation (F-rp3-c7)

**Owns:** C-003 and the relevant part of C-007

**Matrix:** rows R114 and R166

**Depends on:** none (section 11.3 removed the PR-1 edge)

**Amended scope (section 11.2).** Beside the merge-on-read UPDATE repair below, this unit
carries F-rp3-c7: new V3 manifests with no `first_row_id` advance the writer counter by all
added and existing rows (`spec/manifest_list.rs`), which becomes the snapshot's assigned-row
count (`transaction/snapshot.rs`), so rewritten rows carrying a stored `_row_id` are counted as
newly assigned (RePark measured next-row-id 6 where Spark stays 5). Required additions:
the rewrite-aware allocation repair; sequential COW DELETE and UPDATE tests; assertions for the
complete row multiset, `_row_id`, `_last_updated_sequence_number` and next-row-id after every
step; a mutation restoring the count-all-rows allocation that turns both sequential tests red.
RePark re-measures its COW UPDATE lineage statement now and lifts no guard until this unit lands.

#### Implementation

1. Add `_row_id` and `_last_updated_sequence_number` to the V3 merge-on-read UPDATE scan.
2. Capture both lineage arrays before reducing the batch to user columns.
3. Apply the UPDATE expressions to user columns only.
4. Attach the original `_row_id` to every replacement row.
5. Write null for the replacement row's stored `_last_updated_sequence_number`.
6. Let the reader resolve that null from the new data file's sequence number.
7. Keep V1 and V2 behavior unchanged.
8. Keep DV merge and shared-Puffin sibling closure unchanged.

The COW UPDATE implementation in `physical_plan/delete.rs` is the reference behavior. The shared
helpers in `physical_plan/row_lineage.rs` remain the single implementation of lineage attachment.

#### Expected files

- `crates/integrations/datafusion/src/physical_plan/delete.rs`
- `crates/integrations/datafusion/src/physical_plan/row_lineage.rs` only if a shared helper is needed.
- A dedicated V3 merge-on-read lineage integration test.
- `crates/integrations/datafusion/tests/map.md`
- `dev/java-interop/` for Java metadata-column read-back.
- `docs/parity/GAP_MATRIX.md` rows R114 and R166 after the evidence exists.

#### Required tests

| Test | Required assertion |
|---|---|
| One UPDATE | Updated row keeps `_row_id`. Its last-updated sequence advances. Unmatched rows keep both values. |
| Sequential UPDATE | Updating the same row twice keeps one `_row_id`. The last-updated sequence advances twice. |
| Partitioned UPDATE | Lineage stays correct across at least two partitions. |
| Shared Puffin | Updating one row preserves sibling DV blobs and every live row. |
| Commit conflict | Concurrent removal of a referenced data file refuses the commit. No replacement DV becomes live. |
| V2 control | V2 still writes position deletes and has no V3 lineage behavior. |

#### Test adequacy

- Remove the two lineage projections. The first and sequential UPDATE tests must turn red.
- Replace the original row ID with null. The row-ID assertions must turn red.
- Preserve the old last-updated value on the modified row. The sequence assertions must turn red.
- Record every mutation as `N red out of M`.

#### Interop gate

Rust performs two merge-on-read updates on a Java-created V3 table. Java projects the two lineage
metadata columns and verifies stable row IDs plus advancing update sequences. Add the reverse leg
when Java exposes the same merge-on-read shape through the chosen fixture.

#### Exit gate

- Unit, integration, mutation, and Java read-back evidence are green.
- Matrix rows R114 and R166 record only the evidence they own.
- Independent Critic converges with no open S0, S1, or S2 finding.

### PR-4: Close V3 upgrade and maintenance interoperability

**Owns:** C-004 and the relevant part of C-007

**Matrix:** rows R109, R114, R135, R136, and R166

**Depends on:** PR-2 and PR-3

This unit proves composed behavior. It must not substitute for PR-1 through PR-3 regression tests.

#### Required upgrade matrix

| Producer | Upgrade writer | First V3 operation | Required consumer proof |
|---|---|---|---|
| Java V2 | Rust | append | Java and Rust agree on rows, format version, snapshot sequence, and assigned row IDs. |
| Rust V2 | Java | append | Java and Rust agree on the same values. |
| Java V2 with Parquet position deletes | Rust | convert deletes, then UPDATE | Java and Rust agree on live rows. The V3 table has no newly-added Parquet position delete. |
| Rust V2 with Parquet position deletes | Java or Spark | convert or rewrite | Rust reads the same live rows and valid lineage. |

ORC and Avro legacy position-delete conversion is outside this production envelope. Section 6
defines the work if the owner selects it.

#### Required maintenance matrix

Run these actions on V3 tables and compare rows, snapshots, manifests, delete files, and lineage:

1. `RewriteDataFiles` on a current spec.
2. `RewriteDataFiles` after spec evolution.
3. `RewritePositionDeleteFiles` while converting legacy Parquet deletes to DVs.
4. `RewriteManifests` with data and delete manifests.
5. `ExpireSnapshots` after the rewrite sequence.

For `RewriteManifests`, assert that every live file keeps its row ID range. Assert that the table's
`next_row_id` stays above every live assigned range. This mandatory unit exercises ordinary
clustering only. Direct external-manifest input stays in H-3 because RePark does not expose it.

#### Expected files

- `crates/iceberg/tests/interop_*` for upgrade and composed maintenance.
- `dev/java-interop/` runners and Java oracle code.
- `crates/iceberg/src/transaction/rewrite_manifests.rs` if ordinary clustering exposes a range bug.
- `crates/iceberg/src/spec/manifest_list.rs` if ordinary range accounting needs a shared fix.
- `docs/parity/GAP_MATRIX.md` rows named above after evidence exists.
- The touched directories' `map.md` files.

#### Test adequacy

- Every interop runner must discover its expected fixture count.
- Missing environment, oracle, or target bytes must hard-fail this release gate.
- A no-op maintenance action cannot satisfy the test. Assert file sets, snapshot operations, and
  manifest or delete-file counts.
- Mutate each lineage or delete conversion rule separately.

#### Exit gate

- Every upgrade and maintenance cell passes.
- Java and Rust agree on live rows and lineage.
- No test succeeds by skipping its oracle.
- Independent Critic converges with no open S0, S1, or S2 finding.

### PR-5: Implement and prove catalog commit outcomes on Glue and S3 Tables

**Owns:** C-005 and the relevant part of C-007

**Matrix:** rows R110 and R157

**Depends on:** none for the harness, PR-1 through PR-4 for the final run

**Split (section 11.1).** PR-5A is mandatory for V1.0: the narrow commit-transport seams;
offline proof of never-sent, maybe-sent, accepted-but-response-lost, reconciliation success,
reconciliation exhaustion, metadata-only unknown and no blind retry; exactly one offline
CAS/conflict retry test per catalog; one offline test that a permanent authorization denial is
terminal; one credentialed normal smoke per catalog per commit class; exactly one credentialed
accepted-then-response-lost append per catalog (closes row R157's real-catalog claim).
PR-5B holds the full throttle matrix, the counting connector below the AWS SDK retry
middleware, detailed attempt accounting and the exhaustive error cross-product, deferred to the
reliability and multi-writer track. The Phase A ledger, Phase B connector, the 98-cell
cross-product and the throttle rows below describe PR-5B unless PR-5A names them; the
never-sent, accepted-then-lost, conflict, terminal-authorization, reconciliation and
metadata-only rows are PR-5A. Credentialed runs use the existing owner-approved AWS boundary.

#### Required operation partition

Test each catalog with these commit classes:

1. Snapshot append.
2. RowDelta with a V3 DV.
3. RewriteFiles.
4. Schema update.
5. Property update.
6. V2 to V3 upgrade.
7. Snapshot-reference update.

#### Phase A: Freeze the finite outcome ledger

Build a `2 catalogs * 7 operations * 7 outcomes = 98` cell ledger from the operation and outcome
partitions. Mark every cell `SUPPORTED` or `EXEMPT`. An exemption must name the service or operation
constraint that makes the cell unreachable. It needs an owner ruling. Do not run a partial
cross-product and call it the full matrix.

Decode the matching Java Glue and S3 Tables commit paths before fixing expected error kinds. Record
the Java behavior, the AWS SDK retry behavior, and the Rust ruling for never-sent and throttled
failures. A safe terminal result may be parity-correct. Do not invent a retryable classification
only because the request did not leave the client.

#### Phase B: Add deterministic fault seams at both required layers

1. Wrap each catalog's completed commit SDK call behind a narrow internal transport trait.
2. Use this catalog-level seam to stop before send or to call the real client and discard its
   successful response.
3. Use the discard mode for the credentialed accepted-then-response-lost cell. The fault must occur
   after the provider accepts the exact commit and before the catalog receives success.
4. Build a counting fault connector below the AWS SDK retry middleware. It must synthesize exact
   service responses before delegating to the real HTTP connector.
5. Use the connector for throttle, modeled conflict, authorization, and validation responses. A
   throttle test returns the modeled response for a fixed attempt count, then delegates or exhausts
   the configured retry bound.
6. Count catalog commit attempts and underlying HTTP attempts separately. Assert that reconciliation
   never sends a second catalog commit.
7. Keep both seams dependency-free. Any dependency-file change requires separate user approval.

The Glue builder currently constructs a private SDK client. Add internal test constructors for both
the commit transport and HTTP connector. The S3 Tables prebuilt-client hook is not sufficient
because it cannot drop a selected response after a real successful call. A wrapper above `.send()`
is not throttle evidence because it bypasses the SDK retry middleware.

#### Required outcome partition

For each supported operation class, exercise these outcomes:

| Outcome | Required behavior |
|---|---|
| Request is not sent | Return the Phase A parity-approved terminal or retryable result. Make no reconciliation claim. The provider must show no matching commit. |
| Catalog reports a compare-and-swap conflict | Return a conflict. Rebase only when the action's validation contract permits it. |
| Service commits but the response is lost | Return or reconcile `CommitStateUnknown`. Never issue a blind duplicate commit. |
| Service throttles before accepting | Inject below the AWS SDK retry layer. Prove the configured HTTP retry bound and attempt count. After exhaustion, use the Phase A parity-approved result. Do not add a catalog retry that can duplicate a commit. |
| Service returns a terminal authorization or validation error | Do not retry. Do not clean files that might be live. |
| Reload finds the intended snapshot | Reconcile snapshot-producing commits to success. |
| Metadata-only outcome stays ambiguous | Return a typed unknown outcome and require caller reload-and-verify. Never report success without proof. |

#### Harness requirements

- Use a user-approved dedicated AWS account or test boundary, region, Glue database, S3 Tables
  bucket, minimal IAM policy, and cost limit.
- Use dedicated test tables and unique namespaces.
- Never log credentials or credentialed object-store URLs.
- Record catalog request IDs only when they contain no secret.
- Make cleanup additive and scoped. Do not destroy shared namespaces.
- Record the exact catalog, region, operation, injected failure point, and observed result.
- Hard-fail a cell if the transport cannot prove whether the request was stopped, accepted, or
  answered. A timeout without causal injection is not accepted-then-response-lost evidence.
- Record catalog attempts and HTTP attempts as separate fields.

#### Expected files

- `crates/catalog/glue/src/catalog.rs`, `crates/catalog/glue/src/error.rs`, and their
  commit-outcome tests.
- `crates/catalog/s3tables/src/catalog.rs` and its commit-outcome tests.
- A shared or per-catalog counting HTTP connector test helper.
- The catalogs' `map.md` files when routing changes.
- `crates/iceberg/src/transaction/mod.rs` if reconciliation behavior changes.
- Credentialed test runners under `dev/` and their CI workflow entry.
- `docs/parity/GAP_MATRIX.md` rows R110 and R157 after measured evidence exists.

#### Test adequacy

- Bypass the transport's response-loss arm. The accepted-then-lost test must turn red.
- Allow a second commit attempt after response loss. The request-count assertion must turn red.
- Map a never-sent failure to a different kind. Its Phase A parity assertion must turn red.
- Allow the never-sent case to reach the HTTP connector. The zero-HTTP-attempt assertion must turn
  red.
- Move throttle injection above the SDK retry layer. The HTTP-attempt assertion must turn red.
- Change or bypass the configured SDK throttle bound. The exact HTTP-attempt assertion must turn
  red.
- Map conflict to terminal. The conflict and permitted-rebase tests must turn red.
- Map terminal authorization to unknown or retryable. The terminal test must turn red.
- Disable snapshot reconciliation. The intended-snapshot test must turn red.
- Report a metadata-only ambiguous commit as success. The metadata test must turn red.

#### Exit gate

- Glue and S3 Tables pass the full operation and outcome partitions.
- The 98-cell ledger contains no unruled blank cell. Every exemption has an approved reason.
- The deterministic transport proves each failure point and attempt count.
- Any service limitation has a dated, user-approved declaration in the consumer contract.
- Matrix rows R110 and R157 record the measured evidence.
- Independent Critic converges with no open S0, S1, or S2 finding.

### PR-6: Close branch read and commit interoperability

**Owns:** C-006 and the relevant part of C-007

**Matrix:** row R168

**Depends on:** none

**Split (section 11.3).** PR-6A runs first and carries cases 1 through 9 below except the
merge-on-read UPDATE lineage columns; PR-6B adds the merge-on-read UPDATE lineage branch cell
after PR-3 lands.

#### Required cases

1. Rust reads a Java-created branch whose head diverges from `main`.
2. Rust appends to that branch. Java verifies that `main` did not move.
3. Rust performs COW DELETE and UPDATE on that branch.
4. Rust performs merge-on-read DELETE and UPDATE on that branch.
5. A missing branch fails on read and UPDATE.
6. An INSERT-only commit creates a missing branch when the contract permits it.
7. A tag target refuses writes.
8. Retry reloads the named branch head instead of `main`.
9. Java reads a Rust-created branch and verifies its parent and snapshot reference.

#### Exit gate

- All cases pass locally and through Java interop.
- Main and branch file sets are asserted separately.
- Matrix row R168 records the evidence.
- Independent Critic converges with no open S0, S1, or S2 finding.

### PR-7: Production evidence closeout

**Owns:** C-007 and verifies C-001 through C-006

**Depends on:** PR-1 through PR-6

#### Required gates

1. `typos .`
2. `make check`
3. `make check-msrv`
4. `cargo build -p iceberg --no-default-features`
5. `cargo deny check advisories`
6. `make test`
7. The targeted Java interop suites from PR-1 through PR-6.
8. The credentialed Glue and S3 Tables matrix from PR-5.
9. The consumer's V3 scale and statement matrix. RePark owns that execution.

The final report must enumerate every test population. It must identify each CI-only exception.
It must not treat an environment-gated early return as interop evidence.

Gate 9 is an external release prerequisite. It does not authorize fork changes in RePark and it
cannot substitute for any fork-side test.

#### Exit gate

- All mandatory clauses have evidence links.
- All PR Critics converged.
- A fresh bundle-scope Critic attests correctness, edge cases, concurrency, error handling,
  security, data integrity, resource exhaustion, claim gaps, quantifier span, Java parity, and
  format stability.
- No open S0, S1, or S2 finding remains.
- The user approves the production-ready claim.

## 5. Evidence matrix

| Risk | Local regression | Mutation | Java interop | Live catalog |
|---|---:|---:|---:|---:|
| REPLACE adds more rows than it removes | required | required | required | not applicable |
| Same-arity partition evolution during compaction | required | required | required | optional |
| Partition rewrite opens unbounded writers | required | required | not applicable | optional |
| V3 merge-on-read UPDATE changes row identity | required | required | required | required in final AWS run |
| Legacy position deletes survive a V3 upgrade incorrectly | required | required | required | required in final AWS run |
| Ordinary RewriteManifests changes row ranges | required | required | required | optional |
| Unknown commit outcome causes a duplicate commit | mock required | failure injection required | not applicable | required |
| Branch commit moves `main` | required | required | required (PR-6A) | required when refs are enabled there |
| Rewritten rows counted as newly assigned (F-rp3-c7, PR-3) | required | required | required | optional |
| Duplicate commit after response loss (PR-5A) | mock required | failure injection required | not applicable | one credentialed append per catalog |

## 6. Decision-gated fork work

These items do not enter the mandatory PR sequence until the consumer owner selects support. A dated
consumer declaration can keep them outside the V1.0 fork scope. The GAP_MATRIX remains the only
home for capability status.

| Surface | Matrix row | Fork work if selected |
|---|---|---|
| Binary variant over Avro | R88 | Add Avro data read and write plumbing. Add binary exact-byte tests and Java interop. |
| Parquet variant | R88 | Define binary-only versus shredded support. Build the Iceberg-to-Parquet variant schema bridge. Add nested placement tests and cross-engine files. Treat the parquet API as experimental. |
| `unknown` reads | R91 | Materialize optional `unknown` fields as null without reading a physical column. Keep physical writes refused. Add Java metadata and scan interop. |
| Unknown or multi-argument transforms | new matrix row required | Preserve the parsed transform without evaluating an unknown function. Ignore it for partition filtering, refuse selecting it for writes, and add Java metadata plus pruning interop. The ratified V3 spec defines the reader-tolerance rule but no concrete multi-argument transform. |
| Nested or non-primitive write defaults | R92 | Recursively fill missing fields without replacing supplied values. Add struct, list, and map tests. |
| Legacy ORC or Avro position-delete conversion | R136, R118, and R119 | Define one exact upgrade sequence. Prove the V3 read result, conversion result, and unchanged table state on refusal. Add Java interop for each selected format. |
| ORC data | R118 | Finish the selected V3 read and write envelope. Add Java ORC interop. |
| Avro data | R119 | Finish the selected V3 read and write envelope. Include variant only when selected above. |
| Geometry and geography | R89 | Build the type family, format gates, predicates, bounds, Arrow mapping, and interop. A dated declaration currently keeps this outside the mandatory sequence. |
| Encryption | R130 | Build key metadata, encrypted manifest and data IO, key rotation, and interop. A dated declaration currently keeps this outside the mandatory sequence. |

Each selected surface becomes its own STANDARD charter. Do not bundle variant, geospatial types,
encryption, ORC, and Avro into one PR.

## 7. Conditional hardening outside the current envelope

These are real fork risks. They do not block the current single-writer, Glue-and-S3-Tables envelope.

### H-1: Atomic Hadoop metadata publication

**Matrix:** row R167

Two writers can target the same deterministic `vN.metadata.json` path. The loser can overwrite the
winner's bytes around the pointer compare-and-swap. Before multi-writer Hadoop-style tables are
supported, publish through a unique temporary object and an atomic no-overwrite rename or create.
Add a barrier-controlled race test. Assert that the winning pointer always resolves to the winning
metadata bytes.

### H-2: REST vended credential lifecycle

**Matrix:** row R160

Before REST catalogs with vended-only storage access enter production, select credentials per
prefix, retain them after `update_table`, and refresh them before expiry. Test separate metadata and
data prefixes. Test a chained commit through the returned `Table`.

### H-3: Direct external RewriteManifests input

**Matrix:** row R166

If a consumer exposes `RewriteManifestsAction::add_manifest`, prove that an assigned
`first_row_id` cannot overlap a live range or strand `next_row_id`. Otherwise reject that input
before commit. This is a separate STANDARD unit, not PR-4. It needs local range-safety tests and
mutations. A refusal is deliberate fork hardening because Java accepts the input; do not claim
symmetric Java behavior. RePark's ordinary V3 `rewrite_manifests` exercise does not select H-3.

## 8. Work that belongs in RePark, not this fork

The fork plan must not absorb these downstream tasks:

- Repin all Iceberg crates from `33be9a0` to the accepted fork revision.
- Remove the metadata projection and table-listing shims after their fork replacements are consumed.
- Flip the `unknown` and delete-ratio regression tests at the new pin.
- Add the SQL `ALTER TABLE` surface for V2 to V3 upgrade.
- Lift the V3 UPDATE, MERGE, and sequential COW guards after fork evidence passes.
- Run the RePark facade and Spark statement matrix.
- Run the RePark V3 `10^7 x 50` scale campaign.
- Maintain RePark's dated declarations and release registry.

These downstream tasks can depend on fork PRs. They cannot serve as proof that the fork PRs are
correct.

## 9. Per-PR delivery template

Every fork PR uses this evidence block:

```text
Charter clauses:
Matrix rows:
Java methods or bytecode read:
Files changed:
Behavior before:
Behavior after:
Negative cases:
Test command and population:
Mutations, one at a time:
Java interop command and fixture count:
CI-only evidence gap:
Breaking public API change:
Critic attestation:
Open findings and dispositions:
```

The PR body cites matrix rows as `row R<id>`. It never cites GAP_MATRIX line numbers. Capability
status changes only in the owning matrix cell.

## 10. Definition of done

The fork portion of V3 production support is complete only when all statements below are true:

- PR-1 through PR-7 (PR-5A and PR-6A/6B as split) meet their exit gates; PR-5B is not a V1.0 gate.
- Every mandatory clause has a local regression, a load-bearing mutation, and required interop.
- No compaction path can duplicate rows or mislabel partitions silently.
- Every required V3 row rewrite keeps row identity.
- Upgrade and maintenance preserve live rows, lineage, deletes, and snapshot validity.
- Glue and S3 Tables pass the credentialed commit-outcome matrix.
- Unsupported selected-out surfaces fail before writing bytes or committing metadata.
- `typos .`, `make check`, the pre-merge gate, targeted interop, and required live tests are green.
- Every PR has an independent Critic. The bundle has a fresh closing Critic.
- No open S0, S1, or S2 finding remains.
- RePark completes its downstream gate and the user approves the production-ready claim.

Until then, the fork can support a guarded V3 pilot. It cannot support the full production claim.

## 11. Owner approval and binding amendments (2026-09-01)

The owner approved this plan on 2026-09-01 through the RePark orchestrator, after the RePark
review and the fork agent's reply converged. Approval covers the seven-unit shape, the clause
contract in section 2, the evidence matrix in section 5, and the decision-gated table in
section 6, as amended below. Sections 3, 4 and 5 are rewritten to match before PR-6A starts;
this section is the ruling of record until then.

### 11.1 PR-5 splits into PR-5A (mandatory for V1.0) and PR-5B (deferred)

PR-5A owns C-005 for the V1.0 gate:

- Narrow commit-transport seams on the Glue and S3 Tables catalogs.
- Offline proof of never-sent, maybe-sent, accepted-but-response-lost, reconciliation success,
  reconciliation exhaustion, metadata-only unknown, and no blind retry.
- Exactly one offline CAS/conflict retry test per catalog. A single application writer does not
  remove S3 Tables service-maintenance concurrency (`docs/ENGINE_CONTRACT.md`).
- One offline test proving a permanent authorization denial is terminal and never retried.
- One credentialed normal smoke per catalog per commit class (the seven classes in section 4).
- Exactly one credentialed accepted-then-response-lost append per catalog. That closes the
  real-catalog claim in row R157 without the 98-cell ledger.

PR-5B holds the full throttle matrix, the counting connector below the AWS SDK retry
middleware, detailed attempt accounting, and the exhaustive error cross-product. It is
deferred to the reliability and multi-writer track (the 1.9 promise) and does not gate V1.0.

Credentialed runs use the existing owner-approved AWS boundary and IAM statement
(`docs/tier2-aws.md` on the RePark side). This approval does not grant new IAM permissions
or spend beyond that boundary; either needs the owner directly.

### 11.2 PR-3 expands to carry F-rp3-c7

PR-3 owns both V3 row-DML lineage defects, not only the merge-on-read UPDATE omission:

- The rewrite-aware row-allocation repair for F-rp3-c7: new V3 manifests with no
  `first_row_id` advance the writer counter by all added and existing rows
  (`spec/manifest_list.rs`), which becomes the snapshot's assigned-row count
  (`transaction/snapshot.rs`), so rewritten rows carrying a stored `_row_id` are counted as
  newly assigned (RePark measured next-row-id 6 where Spark stays 5).
- Sequential COW DELETE and UPDATE tests.
- Assertions for the complete row multiset, `_row_id`, `_last_updated_sequence_number`, and
  next-row-id after every step.
- The existing merge-on-read UPDATE lineage repair with its mutation table.

RePark's COW UPDATE lineage statement under `V3-COW-1` is stale: the pinned `33be9a0` already
carries fork #243's COW lineage attachment and the four COW lineage tests pass. RePark
re-measures that half now, independent of its repin, and records the result. It does **not**
lift the `V3-COW-1` guard until PR-3 lands the counter repair.

### 11.3 Dependency graph and execution order

The PR-1 → PR-3 edge is removed; the two units touch different producers. PR-6 moves to the
front and splits: PR-6A (Java/Rust branch read, append, COW DML, `main` immobility,
missing-ref and tag refusal) runs immediately because RePark's next repin consumes fork #245
and #249; PR-6B (the merge-on-read UPDATE lineage branch cell) follows PR-3.

Order of record:

1. PR-6A, immediately.
2. In parallel: PR-1, the expanded PR-3, and the PR-5A harness work.
3. PR-2 after PR-1.
4. PR-4 after PR-2 and PR-3.
5. PR-6B after PR-3.
6. PR-5A credentialed execution.
7. PR-7 closeout.
8. PR-5B deferred to the reliability and multi-writer track.

Mechanical overlap between the expanded PR-3 and PR-1 in snapshot accounting is merge
coordination, not a dependency.

### 11.4 Section 6 selections

`unknown` reads (row R91) are **selected**: Spark reads the column as null, the fork work is
null materialization without a physical column read, physical writes stay refused. It becomes
its own STANDARD charter after PR-7 and does not gate V1.0. Geometry and geography, shredded
Parquet variant, and encryption stay outside the mandatory sequence under RePark's dated
declarations. Every other row in section 6 stays unselected until the owner rules.

### 11.5 Process

This file lands on fork `main` through its own PR before PR-6A's first commit. Every unit
keeps the section 9 delivery template and an independent Critic. Section 8 stays the
boundary: RePark's repin (RP-5), shim removal, pin flips and guard lifts are RePark units and
are not evidence for any fork PR.

