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

# ICE-COMMIT-REBASE-DUP-1 — the re-base after a CAS rejection must not append twice

**Date:** 2026-10-15. **Base:** fork `main` `076d5f98` (the RePark pin), branch
`fix/ice-commit-rebase-dup-1`. **Model:** Claude Opus 5.5 (`claude-opus-5-5`). **Card:** RePark
`task/roadmap/mid-term/ice-commit-rebase-dup-1-card-2026-10-15.md`. **Related:** GAP_MATRIX
row R157, [PR-5A ledger](pr5a-catalog-commit-outcomes-ledger.md), the U-9 sketch's cell U-5
(RePark `task/wo/ice-integrity-pins-1/sketch.md` §4.4).

This ledger retires when RePark repins past it and U-9's live drill shows no double append, or
when the owner removes it.

## The ruling

RePark owner ruling 84 (2026-10-15), verbatim:

```
84. Yes, and the card is wider than turning the SDK retry off. The fork
    already re-bases and re-appends after a CAS rejection
    (append.rs:1685). A lost reply, an SDK re-send, a stale rejection, and
    that re-base appends the same rows twice. The drill must run that
    exact sequence. The fork card: after any CAS rejection, refresh and
    check whether the table already holds our snapshot id before
    re-basing; then switch the SDK retry off on the two commit calls. If
    the drill shows a double append, the live leg stops and v2 waits.
```

## Outcome

- **Step 0 (measured).** On `076d5f98` the drill runs the ruling's exact sequence and the rows
  land twice on Glue and on S3 Tables. With the duplicate-file check on (RePark's doors), the
  rows land once but the statement fails with `DataInvalid`. Ruling 84: a double append stops
  the live leg, and v2 waits.
- **Step 1 (fixed).** After any retryable rejection, `do_commit` refreshes and returns the
  refreshed table when it already holds a snapshot id this transaction sent. The drill flips to
  one snapshot and the rows once, on both catalogs. A genuine conflict still re-bases.
- **Step 2 (fixed).** Glue `UpdateTable` and S3 Tables `UpdateTableMetadataLocation` run with
  the SDK retry off for that operation only. After a reset the endpoint sees one commit
  request, and the outcome matches cell U-1 (landed: `Ok`, rows once) or U-4 (absent:
  `CommitStateUnknown`, pre-state rows). Reads keep their retries.
- **Open (Q1).** With the SDK retry off, a lost reply on a normally closed connection (FIN,
  not a reset) is classified never-sent and fails as `Unexpected`, although it landed.

## The drill

`crates/catalog/glue/tests/commit_lost_reply_drill.rs` and
`crates/catalog/s3tables/tests/commit_lost_reply_drill.rs`. Each runs offline, with no AWS.

- **The endpoint.** A test-only HTTP/1.1 server on `127.0.0.1:0`, in the test process. Glue:
  `CreateTable`, `GetTable` and `UpdateTable` (`X-Amz-Target: AWSGlue.*`), with a CAS on the
  `VersionId` the request carries. S3 Tables: `CreateTable`, `GetTable` (`GET /get-table`) and
  `UpdateTableMetadataLocation` (`PUT …/metadata-location`), with a CAS on `versionToken`. A
  stale version gets `ConcurrentModificationException` (HTTP 400) or `ConflictException`
  (HTTP 409). The endpoint logs every request: operation, version sent, answer.
- **The scripts for the commit call.** `ApplyThenReset` applies the commit, then resets the
  connection (`SO_LINGER` 0, so the client reads `ECONNRESET`) before the reply. `DropThenReset`
  reads the commit, does not apply it, and resets. `ApplyThenConflict` applies the commit and
  answers the conflict error. Reads can be reset too (`reset_next_reads`).
- **The client.** The catalog the crate builds through its public builder:
  `GlueCatalogBuilder` / `S3TablesCatalogBuilder` with `uri` / `endpoint_url` set to the
  endpoint, static dummy credentials, `us-east-1`, and the memory storage factory. So the real
  `aws-sdk-glue` 1.158.0 / `aws-sdk-s3tables` 1.62.0 client sends the requests, with its own
  retry policy (`aws-smithy-runtime` 1.12.1, standard mode, 3 attempts).
- **The write.** One `fast_append` of one data file with 3 records, stamped with a snapshot
  property `drill.append`, on an empty table. The duplicate-file check runs as configured
  (`with_check_duplicate`). RePark's append doors use `merge_append` / `fast_append` with the
  default check on; both run the same `validate_duplicate_files` on every attempt.
- **The count.** After the commit, a reload through the endpoint: the snapshots that carry
  `drill.append`, and the rows (the record counts of the live data-file entries in the current
  snapshot's manifests).

Runs used `AWS_PROFILE`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` unset,
`AWS_CONFIG_FILE` and `AWS_SHARED_CREDENTIALS_FILE` pointed at a missing file, and
`AWS_EC2_METADATA_DISABLED=true`.

## Step 0 — measured on `076d5f98` (commit `4ee3c668`)

The SDK re-sends the commit after the reset. The endpoint rejects the re-send as stale. The
fork maps the rejection to a retryable `CatalogCommitConflicts`, `Transaction::commit` retries,
and `do_commit` re-bases on a table that already holds the first send's snapshot.

| Catalog | Script | Duplicate check | Commit requests (version sent → answer) | Result | Snapshots with the append | Rows |
|---|---|---|---|---|---|---|
| Glue | `ApplyThenReset` | off | `1` → applied, reset; `1` → stale; `2` → applied | `Ok` | **2** | **6** |
| Glue | `ApplyThenReset` | on | `1` → applied, reset; `1` → stale | `DataInvalid` "Cannot add files that are already referenced by table" | 1 | 3 |
| Glue | `ApplyThenConflict` | off | `1` → applied, conflict; `2` → applied | `Ok` | **2** | **6** |
| S3 Tables | `ApplyThenReset` | off | `v1` → applied, reset; `v1` → stale; `v2` → applied | `Ok` | **2** | **6** |
| S3 Tables | `ApplyThenReset` | on | `v1` → applied, reset; `v1` → stale | `DataInvalid` (the same message) | 1 | 3 |
| S3 Tables | `ApplyThenConflict` | off | `v1` → applied, conflict; `v2` → applied | `Ok` | **2** | **6** |

Read:

- Without the duplicate check, the rows land twice, silently, on both catalogs.
- With the check (RePark's doors), the rows land once, but the statement fails with
  `DataInvalid` for a commit that landed. A caller that retries the failed statement appends
  the rows a second time.

Full request log, Glue, `ApplyThenReset`, check off (S3 Tables is the same shape with
`UpdateTableMetadataLocation` and `v1` / `v2`):

```
GetTable                       → answered   (Transaction::commit: refresh)
GetTable                       → answered   (update_table: commit base)
UpdateTable  VersionId=1       → applied, connection reset
UpdateTable  VersionId=1       → rejected stale (the SDK's re-send)
GetTable                       → answered   (retry: refresh, then re-base)
GetTable                       → answered   (update_table: commit base)
UpdateTable  VersionId=2       → applied, answered (the second append)
```

With the check on, the log stops after the third `GetTable`: the re-base's duplicate check
raises `DataInvalid` before the second commit request.

## Step 1 — the re-base guard (commit `014136fe`)

`Transaction` keeps `sent_ids`: every snapshot id any attempt handed to `update_table`. It is
never cleared. At the top of each `do_commit`, after the refresh, `rejected_attempt_landed`
(`commit_status.rs`) checks whether the refreshed table holds any sent id. If it does, an
earlier attempt landed: `do_commit` returns the refreshed table and re-applies nothing. If it
does not, the attempt re-bases as before. The first attempt has no sent ids, so nothing changes
for it. Every catalog reaches the check through `Transaction::commit`.

`latest_attempt_snapshot_ids` is unchanged: the unknown-outcome reconciliation still searches
the latest attempt's ids only.

Why every attempt and not only the last one: a retry can fail before its own `update_table`
(a failed refresh). The next retry must still look for the landed attempt.
`landed_attempt_is_found_after_a_failed_refresh` pins that.

| Pin | Asserts | On `076d5f98` | On the fix |
|---|---|---|---|
| `transaction::action::rebase_landed_tests::landed_attempt_rejected_as_conflict_is_not_reapplied` | the first `update_table` lands, then answers a retryable conflict; check off: `Ok`, 1 update call, base + 1 snapshot, our file live once | red: `left: 2, right: 1` (update calls) | green |
| `…::landed_attempt_rejected_as_conflict_commits_under_the_duplicate_check` | the same with the check on: `Ok` | red: `DataInvalid => Cannot add files that are already referenced by table, files: test/ours.parquet` | green |
| `…::landed_attempt_is_found_after_a_failed_refresh` | the same, and the first refresh after the conflict fails with a retryable error | red: `left: 2, right: 1` | green |
| `…::other_writer_wins_and_our_append_rebases_and_lands_once` | another writer lands first; ours re-bases: 2 update calls, base + 2 snapshots, both files live, ours once | green | green |
| Glue drill `glue_sdk_resend_rejected_stale_lands_once_without_the_duplicate_check` | commits `[1 → applied, reset; 1 → stale]`, `Ok`, 1 snapshot, 3 rows | red: `left: [(Some("1"), AppliedThenReset), (Some("1"), RejectedStale), (Some("2"), Answered)]` | green |
| Glue drill `…_under_the_duplicate_check` | the same with the check on | red: `left: Err(DataInvalid), right: Ok(())` | green |
| Glue drill `glue_landed_commit_answered_conflict_lands_once` | commits `[1 → applied, conflict]`, `Ok`, 1 snapshot, 3 rows | red: `left: [(Some("1"), AppliedThenConflict), (Some("2"), Answered)]` | green |
| S3 Tables drill, the same three | the same, with `v1` | red: `left: [(Some("v1"), AppliedThenReset), (Some("v1"), RejectedStale), (Some("v2"), Answered)]`; `left: Err(DataInvalid), right: Ok(())`; `left: [(Some("v1"), AppliedThenConflict), (Some("v2"), Answered)]` | green |

Request log on the fix, Glue, `ApplyThenReset` (check on or off):

```
GetTable                       → answered
GetTable                       → answered
UpdateTable  VersionId=1       → applied, connection reset
UpdateTable  VersionId=1       → rejected stale (the SDK's re-send)
GetTable                       → answered   (retry: refresh; it holds our snapshot → Ok, no re-base)
```

## Step 2 — the SDK retry off on the two commit calls (commit `aa0decf8`)

`LiveGlueCommitTransport::send_update_table` and
`LiveS3TablesCommitTransport::send_update_metadata_location` send the call through
`customize().config_override(aws_sdk_*::config::Builder::default().retry_config(RetryConfig::disabled()))`.
`RetryConfig::disabled()` is standard mode with one attempt. The override applies to that one
operation. The client's retry policy is unchanged, so every read keeps its retries. Glue's
staged replace publish (`catalog/replace_publish.rs`) sends through the same transport and
inherits the switch. The SDK has a per-operation switch, so no client-wide change was needed.

A reset after the commit request is now a `DispatchFailure` with an IO connector error. The
fork classifies it `MaybeSent`, maps it to `CommitStateUnknown`, never retries it, and
reconciles it by refresh.

| Pin (per catalog, `tests/commit_lost_reply_drill.rs`) | Asserts | On `076d5f98` | On the fix |
|---|---|---|---|
| `*_lost_reply_is_not_resent_and_reconciles_as_landed` (cell U-1) | `ApplyThenReset`, check on: one commit request `[1 → applied, reset]`; `Ok`; 1 snapshot; 3 rows | red: `left: [(Some("1"), AppliedThenReset), (Some("1"), RejectedStale)]` (S3 Tables: `v1`) | green |
| `*_lost_unapplied_commit_is_unknown_and_keeps_the_pre_state` (cell U-4) | `DropThenReset`: one commit request `[1 → dropped, reset]`; `Err(CommitStateUnknown)`; 0 snapshots; 0 rows | red: `left: [(Some("1"), DroppedThenReset), (Some("1"), Answered)]`: the SDK re-sent the dropped commit and the re-send landed | green |
| `*_landed_commit_answered_conflict_lands_once` | the step-1 guard through the real SDK, unchanged by step 2 | red (step 1 table) | green |
| `*_reads_keep_their_sdk_retry` | the first `GetTable` is reset; the load succeeds; reads `[reset, answered]` | green | green |

On step 1 alone (the SDK retry still on), U-1 and U-4 are red the same way: the endpoint sees
the SDK's re-send.

Request logs on the fix (S3 Tables has the same shape):

```
U-1, ApplyThenReset                          U-4, DropThenReset
GetTable              → answered             GetTable              → answered
GetTable              → answered             GetTable              → answered
UpdateTable  1        → applied, reset       UpdateTable  1        → dropped, reset
GetTable              → answered             GetTable              → answered
  (reconciliation: our snapshot is there:      (reconciliation: our snapshot is absent:
   Ok, rows once)                               CommitStateUnknown, rows as before)
```

## Finding — a lost reply on a closed connection is "never sent" once the SDK retry is off

Probe only, not committed: the same drill with the connection closed normally (FIN) instead
of reset. hyper reports `IncompleteMessage`. `aws-smithy-http-client` 1.2.0 maps that to
`ConnectorError::other(…, Some(ErrorKind::TransientError))` (`src/client.rs`,
`to_connector_error`). The fork's `classify_commit_send_disposition` (Glue `src/error.rs`, S3
Tables `src/commit_transport.rs`) maps every `DispatchFailure` with `is_other()` to
`NeverSent`.

| Code | Glue, `ApplyThenReset` with FIN, check on | S3 Tables, the same |
|---|---|---|
| `076d5f98` | commits `[1 → applied; 1 → stale]`; `Err(DataInvalid)`; 1 snapshot; 3 rows | the same with `v1` |
| step 1 (SDK retry on) | commits `[1 → applied; 1 → stale]`; `Ok`; 1 snapshot; 3 rows | the same with `v1` |
| step 2 (SDK retry off) | commits `[1 → applied]`; `Err(Unexpected)` "… before the update request was sent"; no reconciliation read; 1 snapshot; 3 rows | the same with `v1` |

So with step 2, a commit that landed and lost its reply on a closed connection fails as a
definite, never-sent error. No rows are duplicated by the fork. A caller that retries the
statement appends the rows again. Under step 1 alone, the SDK re-send and the guard resolve
the same case to `Ok`. This is the hand-back's question Q1. The unit did not change the
classifier.

## Residue (not changed here)

- The unknown-outcome reconciliation still searches only the latest attempt's snapshot ids.
- A commit found by the guard fires no `CreateSnapshotEvent`, as the reconciliation path.
- Glue `CreateTable` and the S3 Tables create path's own `UpdateTableMetadataLocation` keep
  the SDK retry. A re-sent create fails as a plain error after it landed. No rows are involved.
- The guard reads the refreshed table once. S3 Tables read-after-write consistency right after
  a commit is unmeasured (U-9 sketch §10).

## RePark never worse

In the RePark clone (no file edited), the four crates ran without and with a command-line patch
override that points `iceberg`, `iceberg-datafusion`, `iceberg-catalog-glue`,
`iceberg-catalog-s3tables` and `iceberg-storage-opendal` at this branch (`--config
'patch.crates-io.<crate>.path="…"'`). `cargo tree -i iceberg` resolved to the path. Only the six
fork packages changed source in the lock; no third-party version moved. RePark's `Cargo.lock`
was restored afterwards.

| Run | `cargo test --no-fail-fast -p repark-iceberg -p repark-core -p repark-spark -p repark-sql` |
|---|---|
| Fork `076d5f98` (the pin) | exit 0: 5905 passed, 0 failed, 11 ignored |
| Fork `aa0decf8` (this branch) | exit 0: 5905 passed, 0 failed, 11 ignored; no test green before and red after |
| P17 `append_midflight_commit_conflict_retries_and_both_appends_land` on `aa0decf8` | exit 0, green |

## Mutations

Each was applied to `mod.rs`, run with `cargo test -p iceberg --lib
transaction::action::rebase_landed_tests`, and reverted (`git diff` empty afterwards).

| Mutation | Result |
|---|---|
| Check only the previous attempt's ids (`mem::take` of `latest_attempt_snapshot_ids`) | exit 101: `landed_attempt_is_found_after_a_failed_refresh` red, `left: 2, right: 1` |
| Never re-base after a sent attempt (`… \|\| !self.sent_ids.is_empty()`) | exit 101: `other_writer_wins_and_our_append_rebases_and_lands_once` red, `left: 1, right: 2` |
| No guard (`076d5f98`) | the step-1 table's red column |

## Gates (fork, on `aa0decf8`)

| Command | Exit |
|---|---|
| `cargo test --no-fail-fast -p iceberg -p iceberg-catalog-glue -p iceberg-catalog-s3tables` | 101: 5122 passed, 14 failed, 21 ignored. The 14 are `tests/glue_catalog_test.rs`, which needs the Docker moto endpoint on `127.0.0.1:5001`; they fail the same way on `076d5f98` (5116 passed, 14 failed). Every test green on `076d5f98` is green here, except the six step-0 tests that steps 1 and 2 replaced. |
| `cargo fmt --all -- --check` | 0 |
| `cargo clippy --all-targets --all-features --workspace -- -D warnings` | 0 |
| `make check-rust-file-size`, `make check-comment-blocks`, `make check-agent-artifacts`, `make check-matrix-anchors` | 0 |
| `typos` on the changed paths | 0 |
