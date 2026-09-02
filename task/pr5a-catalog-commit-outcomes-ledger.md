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

# PR-5A ledger — catalog commit-outcome conformance (mandatory half)

Plan: `task/iceberg-v3-production-work-plan-2026-09-01.md` section 11.1 / clause C-005.
Matrix: row R110, row R157.
Base: `00cdde0`. Branch: `repark/pr5a`.
Docker `make test` legs excused (Docker unavailable).

## 1. Shared commit path

All seven required commit classes share one catalog CAS send:

| Class | Transaction action | Path |
|---|---|---|
| Snapshot append | `fast_append` | Glue `update_table` / S3 Tables `cas_update_metadata_location` |
| RowDelta with a V3 DV | `row_delta` + Puffin DV | same |
| RewriteFiles | `rewrite_files` | same |
| Schema update | `update_schema` | same |
| Property update | `update_table_properties` | same |
| V2 to V3 upgrade | `upgrade_table_version` | same |
| Snapshot-reference update | `manage_snapshots().create_branch` | same |

Proved once per path, with the class named in the never-sent and maybe-sent loops.

## 2. Phase A outcome ledger (PR-5A cells)

| Outcome | Java 1.10.0 | AWS SDK | Rust ruling |
|---|---|---|---|
| Never sent (`SdkError::ConstructionFailure`, user/other dispatch) | `doCommit` catch `RuntimeException` is not `AwsServiceException`, so it calls `checkCommitStatus`. Absent metadata location converts strict FAILURE to UNKNOWN in production non-strict mode. | Request is not written. SDK retries do not apply. | **Terminal `Unexpected`**. No reconciliation claim. A safe terminal result is parity-correct. Do not invent retryable because the request did not leave. |
| Maybe sent (timeout, io dispatch, response error) | Not `AwsServiceException`, or SDK `RetryDetector.retried()` is true: `checkCommitStatus` then `CommitStateUnknownException`. | SDK may retry throttle/5xx below this seam (PR-5B). | **`CommitStateUnknown`**, not retryable. No second catalog commit. |
| Accepted then response lost | Same as maybe-sent after a successful persist. | Live send returns Ok; discard wrapper reports timeout. | Publish the pointer, feed the classifier a timeout, **`CommitStateUnknown`**, then `Transaction::commit` reconciles. Attempt count = 1. |
| CAS conflict | `handleAWSExceptions`: `ConcurrentModificationException` → `CommitFailedException`. REST 409 → `CommitFailedException`. | Modeled service error. | **`CatalogCommitConflicts` retryable**. Rebase where validation permits (fast append default validate is a no-op). S3 Tables service-side maintenance is a concurrent committer (`ENGINE_CONTRACT.md` §8). |
| Permanent authorization | Glue `AccessDeniedException` → `ForbiddenException`. REST `DefaultErrorHandler` 403 → `ForbiddenException`. | Glue `UpdateTableError` has no modeled AccessDenied; code `AccessDeniedException` is Unhandled. S3 Tables `ForbiddenException` is modeled. | **Terminal `Unexpected`**. Not retried. Staged metadata is not cleaned. |
| Reload finds intended snapshot | `checkCommitStatus` SUCCESS swallows persist failure. | n/a | `Transaction::reconcile_unknown_commit_outcome` → `Ok`. |
| Reload never confirms | Non-strict FAILURE → UNKNOWN. | n/a | Original `CommitStateUnknown` surfaces. |
| Metadata-only ambiguous | Java location check can cover these. | n/a | No `AddSnapshot` evidence. Typed unknown. Never success. Named divergence. |

Throttle matrix and HTTP-attempt counting are PR-5B.

## 3. Java decode

Command: `dev/java-interop/run-pr5a-catalog-commit-decode.sh`
Fixture count: **12 needles** (hard-fail if different).

`javap -c -p` iceberg-aws 1.10.0 `GlueTableOperations`:

- `doCommit`: writes metadata, `persistGlueTable` → Glue `updateTable`. Catch `CommitFailedException` rethrow. Catch `RuntimeException`: if `AwsServiceException` and `RetryDetector.retried()` is false, skip `checkCommitStatus`; else reconcile. Status SUCCESS continue; FAILURE → `CommitFailedException`; UNKNOWN → `CommitStateUnknownException`.
- `handleAWSExceptions`: ConcurrentModification → CommitFailed; AlreadyExists; EntityNotFound → NotFound; AccessDenied → Forbidden; ValidationException; 5xx rethrow.
- `RetryDetector.publish`: `CoreMetric.RETRY_COUNT > 0`.

iceberg-core 1.10.0 REST:

- `ErrorHandlers$CommitErrorHandler.accept` lookupswitch: 404 NoSuchTable, 409 CommitFailed, 500/502/503/504 CommitStateUnknown. 403 falls through.
- `ErrorHandlers$DefaultErrorHandler.accept` lookupswitch: 403 ForbiddenException.

iceberg-aws 1.10.0 has **no** `S3TablesTableOperations`. Java S3 Tables is REST-backed.

## 4. Proposition table

| Id | Proposition | Pin |
|---|---|---|
| P1 | Never-sent is terminal Unexpected, not unknown | `never_sent_is_terminal_for_every_commit_class_on_the_shared_*_path` |
| P2 | Maybe-sent is CommitStateUnknown, attempts stay at one extra seed | `maybe_sent_is_unknown_without_a_second_commit_for_every_commit_class` |
| P3 | Accepted-then-lost append reconciles, attempts=1 | `accepted_then_lost_append_reconciles_without_a_duplicate_commit` |
| P4 | Exhaustion stays unknown | `accepted_then_lost_without_the_snapshot_stays_unknown` |
| P5 | Metadata-only lost is unknown, never Ok | `metadata_only_accepted_then_lost_is_typed_unknown_never_success` |
| P6 | CAS conflict rebases when validation permits | `cas_conflict_rebases_when_the_append_validation_contract_permits` |
| P7 | Authorization is terminal, staged files remain | `permanent_authorization_denial_is_terminal_and_does_not_clean_staged_files` |

## 5. Mutations (`N red out of M`)

Command: `cargo test -p iceberg-catalog-glue --lib --offline --locked` (M=41). Restore + `touch` after each.

| Mutation | Result | Tests that went red |
|---|---|---|
| M1 Bypass response-loss arm (`AcceptThenLose` → `Success`) | **2 red out of 41** | `accepted_then_lost_append_reconciles_without_a_duplicate_commit`, `metadata_only_accepted_then_lost_is_typed_unknown_never_success` |
| M2 Accepted-lost maps retryable conflict | **2 red out of 41** | same pair (second catalog attempt / kind) |
| M3 Never-sent → `CommitStateUnknown` | **2 red out of 41** | `never_sent_is_terminal_for_every_commit_class_on_the_shared_glue_path`, `test_never_sent_sdk_error_maps_terminal_unexpected` |
| M4 Conflict not retryable | **2 red out of 41** | `cas_conflict_rebases_when_the_append_validation_contract_permits`, `test_concurrent_modification_stays_retryable_conflict` |
| M5 AccessDenied → `CommitStateUnknown` | **2 red out of 41** | `permanent_authorization_denial_is_terminal_and_does_not_clean_staged_files`, `test_access_denied_is_terminal_not_unknown` |
| M6 Disable snapshot reconciliation (`Transaction::commit` returns the unknown error) | **1 red out of 41** | `accepted_then_lost_append_reconciles_without_a_duplicate_commit` |
| M7 Metadata-only unknown returns `Ok` | **2 red out of 41** | `metadata_only_accepted_then_lost_is_typed_unknown_never_success`, `maybe_sent_is_unknown_without_a_second_commit_for_every_commit_class` |

S3 Tables command: `cargo test -p iceberg-catalog-s3tables --lib --offline --locked` (M=39):

| Mutation | Result |
|---|---|
| S3 M1 Bypass loss | **2 red out of 39** |
| S3 M3 Never-sent kind | **2 red out of 39** |
| S3 M4 Conflict terminal | **2 red out of 39** |
| S3 M5 Forbidden → unknown | **2 red out of 39** |

Restore-green: glue 41 passed, s3tables 39 passed. Files restored from `.bak` + `touch`.

## 6. Credentialed runner (not executed)

Script: `dev/pr5a-catalog-commit-outcomes.sh`

Required env:

- `ICEBERG_PR5A_CREDENTIALED=1`
- `ICEBERG_PR5A_GLUE_WAREHOUSE`
- `ICEBERG_PR5A_S3TABLES_BUCKET_ARN`
- an AWS credential source (`AWS_ACCESS_KEY_ID` or `AWS_PROFILE` or container/web-identity)

The script hard-fails if the armed flag is set and any of those is absent. It does not print credentials or object-store URLs. It records `catalog_attempts_field=catalog_commit_attempts` and `http_attempts_field=unavailable:pr5b`.

Rust tests (armed only):

- `credentialed_glue_commit_class_smokes_and_one_accepted_then_lost_append`
- `credentialed_s3tables_commit_class_smokes_and_one_accepted_then_lost_append`

Unique namespaces `pr5a{millis}`. Cleanup drops the table and namespace. One discarding-transport append per catalog.

## 7. Gates

| Command | Exit |
|---|---|
| `make check` | 0 |
| `cargo test -p iceberg-catalog-glue --lib --locked` | 0 (41 passed) |
| `cargo test -p iceberg-catalog-s3tables --lib --locked` | 0 (39 passed) |
| `dev/java-interop/run-pr5a-catalog-commit-decode.sh` | 0 (12 needles) |
| `make check-matrix-anchors` | 0 (inside `make check`) |
| `cargo test -p iceberg --lib --locked` | 0 (3559 passed, 1 ignored) |

Docker `make test` legs excused.

## 8. Section 9 delivery template

```text
Charter clauses: C-005, C-007 (test adequacy)
Matrix rows: row R110, row R157
Java methods or bytecode read: GlueTableOperations.doCommit, handleAWSExceptions, RetryDetector.retried; ErrorHandlers$CommitErrorHandler (409/5xx); ErrorHandlers$DefaultErrorHandler (403). No S3TablesTableOperations in iceberg-aws 1.10.0.
Files changed: crates/catalog/glue/src/{catalog.rs,commit_transport.rs,commit_outcome_tests.rs,lib.rs,error.rs}, crates/catalog/s3tables/src/{catalog.rs,commit_transport.rs,commit_outcome_tests.rs,lib.rs}, maps, GAP_MATRIX, task ledger/todo, dev runners, scripts/check_rust_file_size.py ceilings
Behavior before: Glue/S3 Tables classified SdkError at `.send()` with no test seam to stop, discard, or model a service response
Behavior after: a narrow GlueCommitTransport / S3TablesCommitTransport wraps the completed commit SDK call; offline proofs cover the PR-5A outcome partition
Negative cases: never-sent, maybe-sent, accepted-then-lost, exhaustion, metadata-only unknown, CAS conflict, authorization denial
Test command and population: cargo test -p iceberg-catalog-glue --lib --locked (41); cargo test -p iceberg-catalog-s3tables --lib --locked (39)
Mutations, one at a time: M1 2/41, M2 2/41, M3 2/41, M4 2/41, M5 2/41, M6 1/41, M7 2/41 (glue); matching S3 Tables knobs
Java interop command and fixture count: dev/java-interop/run-pr5a-catalog-commit-decode.sh — 12 needles
CI-only evidence gap: credentialed Glue/S3 Tables smokes and the accepted-then-lost append (ICEBERG_PR5A_CREDENTIALED=1). HTTP attempt counting is PR-5B.
Breaking public API change: none
Critic attestation: pending independent Critic
Open findings and dispositions: none from this Actor
```
