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
- **Q1 (fixed, commit `3aeb2928`).** With the SDK retry off, a lost reply on a normally closed
  connection (FIN, not a reset) was classified never-sent and failed as `Unexpected`, although
  it landed. The classifier now treats it as maybe-sent, so it reaches the reconciliation: U-1
  and U-4 hold for a close as well as a reset. A refused commit connection is never-sent and is
  not reconciled.
- **Owner ruling 115 (fixed, commit `9830e241`).** Every outcome the owner lists as ambiguous now
  reaches the reconciliation: a 5xx reply, modeled or not, and every dispatch failure but a user
  error or a refused connect. The table is in the ruling-115 section below.

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
  answers the conflict error. `ApplyThenClose` and `DropThenClose` do the same as the reset
  scripts but close the connection normally (FIN), with no reply. Reads can be reset too
  (`reset_next_reads`). In the one-request-per-connection mode, every reply carries
  `connection: close`; `refuse_the_commit_connection(2)` closes the listener before it answers
  the second read after arming (the commit base read), so the commit's connect is refused. The
  listener comes back one second later, so a reconciliation read, if any, reaches the log.
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

## Q1 — a lost reply on a closed connection was "never sent" once the SDK retry was off

The ruling on Q1, quoted: "Q1 (a), orchestrator, provisional pending the owner".

**The measurement that raised it** (an uncommitted probe on `aa0decf8`: the U-1 drill with the
connection closed normally instead of reset, check on). hyper reports `IncompleteMessage`.
`aws-smithy-http-client` 1.2.0 maps that to `ConnectorError::other(…,
Some(ErrorKind::TransientError))` (`src/client.rs` and `src/hyper_legacy.rs`,
`to_connector_error`). The fork's `classify_commit_send_disposition` mapped every
`DispatchFailure` with `is_other()` to `NeverSent`.

| Code | Glue, `ApplyThenReset` with FIN, check on | S3 Tables, the same |
|---|---|---|
| `076d5f98` | commits `[1 → applied; 1 → stale]`; `Err(DataInvalid)`; 1 snapshot; 3 rows | the same with `v1` |
| step 1 (SDK retry on) | commits `[1 → applied; 1 → stale]`; `Ok`; 1 snapshot; 3 rows | the same with `v1` |
| step 2 (SDK retry off) | commits `[1 → applied]`; `Err(Unexpected)` "… before the update request was sent"; no reconciliation read; 1 snapshot; 3 rows | the same with `v1` |

**The signals the SDK gives** (`aws-smithy-runtime-api` 1.14.0 `ConnectorError`): the kind
(`is_io`, `is_timeout`, `is_user`, `is_other`), `as_other() -> Option<ErrorKind>`, and
`connection_metadata()`. The connection status `NeverConnected` exists but has no public
accessor; `connection_metadata()` is `None` for "never connected" and for "unknown" alike.

- A narrower signal than `is_other()` exists for the closed connection: `as_other() ==
  Some(ErrorKind::TransientError)`. In `aws-smithy-http-client` 1.2.0 only `IncompleteMessage`
  produces it, in both client implementations. `ErrorKind` is reached as
  `aws_config::retry::ErrorKind`; `aws-config` is already a dependency of both crates.
- A refused connect has no typed signal. The SDK maps every `hyper_util` connect error to
  `ConnectorError::io` (`src/client.rs`, `downcast_error`), the same kind as a reset after the
  request. The connect error keeps the OS error in its source chain
  (`hyper_util::client::legacy::Error` → `ConnectError` → `std::io::Error`), so the fork reads
  `std::io::ErrorKind::ConnectionRefused` there. `ECONNREFUSED` comes only from `connect`, so the
  request was never written. With the SDK retry off there is exactly one attempt.
- DNS failures and connect timeouts were not drilled: the drill reaches `127.0.0.1` only. Both
  stay `MaybeSent`. The timeout error type (`HttpTimeoutError`) is crate-private, so no typed
  signal separates a connect timeout from a read timeout.

**The change** (commit `3aeb2928`, Glue `src/error.rs` and S3 Tables `src/commit_transport.rs`):

| `DispatchFailure` | Before (`cb9b46f3`) | After |
|---|---|---|
| `as_other() == Some(TransientError)` (closed connection) | NeverSent | **MaybeSent** |
| `is_io()` with `ConnectionRefused` in the source chain | MaybeSent | **NeverSent** |
| any other `is_user()` / `is_other()` | NeverSent | NeverSent |
| any other `is_io()` / `is_timeout()` | MaybeSent | MaybeSent |

The brief said a refused connect "stays" NeverSent. On `cb9b46f3` it was MaybeSent: the
refused-connection pin is red there (`left: 3, right: 2` reads, `CommitStateUnknown`). The
change makes it NeverSent, as the brief's rule asks.

**The cells** (per catalog, `tests/commit_lost_reply_drill.rs`, check on):

| Pin | Asserts | Helper at `cb9b46f3` | On `3aeb2928` |
|---|---|---|---|
| `*_lost_reply_on_a_closed_connection_reconciles_as_landed` (U-1, FIN) | `ApplyThenClose`: one commit request; `Ok`; 1 snapshot; 3 rows | red: `left: Err(Unexpected), right: Ok(())` | green |
| `*_unapplied_commit_on_a_closed_connection_is_unknown_and_keeps_the_pre_state` (U-4, FIN) | `DropThenClose`: one commit request; `Err(CommitStateUnknown)`; 0 snapshots; 0 rows | red: `left: Err(Unexpected), right: Err(CommitStateUnknown)` | green |
| `*_refused_commit_connection_is_never_sent_and_not_reconciled` | no commit request; 2 reads; `Err(Unexpected)`; 0 snapshots; 0 rows | red: `left: 3, right: 2` (a reconciliation read; the outcome was `CommitStateUnknown`) | green |

Request logs on `3aeb2928` (S3 Tables has the same shape with `UpdateTableMetadataLocation v1`):

```
U-1, ApplyThenClose            U-4, DropThenClose             refused commit connection
GetTable → answered            GetTable → answered            GetTable → answered
GetTable → answered            GetTable → answered            GetTable → answered (listener closed)
UpdateTable 1 → applied, FIN   UpdateTable 1 → dropped, FIN   (UpdateTable connect refused)
GetTable → answered (found)    GetTable → answered (absent)   (no reconciliation read)
Ok, rows 3                     CommitStateUnknown, rows 0     Unexpected, rows 0
```

On `cb9b46f3` the FIN logs stop after the commit request (no reconciliation read), and the
refused log has a third `GetTable` (the reconciliation).

Unit pins: Glue `error::tests::test_incomplete_message_classifies_maybe_sent_and_unclassified_other_stays_never_sent`
and `…::test_refused_connection_classifies_never_sent_and_reset_stays_maybe_sent`; S3 Tables
`test_commit_send_disposition_split` gains the same four cases.

## Residue (not changed here)

- The unknown-outcome reconciliation still searches only the latest attempt's snapshot ids.
- 2026-10-16, owner ruling 127 (commit `9a9f91baf920`): answered. The unknown-outcome read-back now
  searches `sent_ids`, every snapshot id this transaction has sent. Pin:
  `*_a_late_landing_attempt_is_found_by_the_unknown_outcome_read_back`.
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
| Fork `3aeb2928` (Q1 (a)) | exit 101: 5904 passed, 1 failed, 11 ignored. The red test is `repark-core` `microbatch::lifecycle_tests::two_sessions_on_one_sink_wait_for_the_scope`: its 10-second bound expired (`the query ends within the bound: Elapsed(())`). Re-run on the same tree: the test alone 8 of 8 green with the override and 8 of 8 without, and its module `microbatch::lifecycle_tests` green with the override. `crates/iceberg`, `crates/integrations` and `crates/storage` are byte-identical between `aa0decf8` (where it was green with the override) and `3aeb2928`, and the test runs on the memory catalog, which never reaches the changed classifiers. Read: a timing flake under load (the 15-minute load average was 12.8 just after the run), not a regression. |
| Fork `3aeb2928`, full re-run | exit 0: 5905 passed, 0 failed, 11 ignored; no test green before and red after |
| P17 on `3aeb2928` | exit 0, green |

## Mutations

Each was applied to `mod.rs`, run with `cargo test -p iceberg --lib
transaction::action::rebase_landed_tests`, and reverted (`git diff` empty afterwards).

| Mutation | Result |
|---|---|
| Check only the previous attempt's ids (`mem::take` of `latest_attempt_snapshot_ids`) | exit 101: `landed_attempt_is_found_after_a_failed_refresh` red, `left: 2, right: 1` |
| Never re-base after a sent attempt (`… \|\| !self.sent_ids.is_empty()`) | exit 101: `other_writer_wins_and_our_append_rebases_and_lands_once` red, `left: 1, right: 2` |
| No guard (`076d5f98`) | the step-1 table's red column |

## Gates (fork, on `aa0decf8`)

The gates on `3aeb2928` follow this table.

| Command | Exit |
|---|---|
| `cargo test --no-fail-fast -p iceberg -p iceberg-catalog-glue -p iceberg-catalog-s3tables` | 101: 5122 passed, 14 failed, 21 ignored. The 14 are `tests/glue_catalog_test.rs`, which needs the Docker moto endpoint on `127.0.0.1:5001`; they fail the same way on `076d5f98` (5116 passed, 14 failed). Every test green on `076d5f98` is green here, except the six step-0 tests that steps 1 and 2 replaced. |
| `cargo fmt --all -- --check` | 0 |
| `cargo clippy --all-targets --all-features --workspace -- -D warnings` | 0 |
| `make check-rust-file-size`, `make check-comment-blocks`, `make check-agent-artifacts`, `make check-matrix-anchors` | 0 |
| `typos` on the changed paths | 0 |

## Gates (fork, on `3aeb2928`)

| Command | Exit |
|---|---|
| `cargo test --no-fail-fast -p iceberg-catalog-glue -p iceberg-catalog-s3tables --test commit_lost_reply_drill` with the helper at `cb9b46f3` | 101: per catalog, the FIN U-1, FIN U-4 and refused pins red, the other 4 green |
| the same on `3aeb2928` | 0: 7 + 7 passed |
| `cargo test --no-fail-fast -p iceberg-catalog-glue -p iceberg-catalog-s3tables --lib` | 0: 71 + 62 passed |
| `cargo test --no-fail-fast -p iceberg -p iceberg-catalog-glue -p iceberg-catalog-s3tables` | 101: 5130 passed, 14 failed, 21 ignored; the 14 are the Docker-only `tests/glue_catalog_test.rs`. Every test green on `cb9b46f3` is green here. |
| `cargo fmt --all -- --check` | 0 |
| `cargo clippy --all-targets --all-features --workspace -- -D warnings` | 0 |
| `make check-rust-file-size`, `make check-comment-blocks`, `make check-agent-artifacts`, `make check-matrix-anchors` | 0 |
| `typos` on the changed paths | 0 |

## Owner ruling 115, 2026-10-16

The ruling, verbatim:

```
115: (a), keep. A lost reply on a closed connection is "may have been
     sent": read the table back for our own snapshot id before any
     retry. The card lists the ambiguous outcomes (connection closed
     before reply, timeout, server error or unknown) against the
     definitive ones (a CAS conflict, which takes the re-base path with
     the same check). The id checked is the one chosen before sending.
```

### What disagreed, measured before the change

Both shapes were drilled through the real SDK client on both catalogs, with the classifier at
`3aeb2928`:

- **A 5xx reply with a code the SDK does not model** arrives as `ServiceError(Unhandled { code:
  Some("InternalFailure") })` with reply status 500. The classifier said ResponseReceived, and the
  service-error mapper's last arm said `Unexpected`, so it was not reconciled.
- **An unparsable reply** arrives as `DispatchFailure(ConnectorError { kind: Other(None), source:
  hyper_util::client::legacy::Error(SendRequest, hyper::Error(Parse(Status))) })`, after the
  endpoint had read the whole request. The classifier's `is_other()` arm said NeverSent, so the
  commit failed with `Unexpected` "… before the update request was sent".

Does `aws-smithy-http-client` 1.2.0 produce `other(None)` only before a request byte is written?
No. It builds the shape in seven places:

- `src/client.rs:579` and `src/hyper_legacy.rs:374`: `HttpResponse::try_from(response)` failed,
  after a response arrived.
- `src/client.rs:642` and `src/hyper_legacy.rs:426`: `to_connector_error`'s last line, for a
  hyper error it does not recognise. A response parse error is one of these, as measured.
- `src/client.rs:614` and `src/hyper_legacy.rs:399` (`downcast_error`, "we have no idea"): any
  other error.

Only `src/hyper_legacy.rs:355` (the request failed to convert to an HTTP 0.2 request) is
provably before the write. So the shape is ambiguous, and it now reconciles. The change needed only
the two classifiers. The service-error mappers are unchanged: a 5xx now reaches the classifier's
`MaybeSent` before the mapper runs.

### The change (commit `9830e241`)

`classify_commit_send_disposition`, Glue `src/error.rs` and S3 Tables `src/commit_transport.rs`,
now takes `SdkError<E, HttpResponse>` and reads the reply status:

| `SdkError` | `3aeb2928` | `9830e241` |
|---|---|---|
| `ConstructionFailure` | NeverSent | NeverSent |
| `DispatchFailure`, `is_user()` or `ConnectionRefused` in the source chain | NeverSent | NeverSent |
| `DispatchFailure`, `other(Some(TransientError))` | MaybeSent | MaybeSent (now through the general rule) |
| `DispatchFailure`, any other `is_other()` | NeverSent | **MaybeSent** |
| `DispatchFailure`, `is_io()` / `is_timeout()` | MaybeSent | MaybeSent |
| `ServiceError`, reply status 5xx | ResponseReceived | **MaybeSent** |
| `ServiceError`, any other status | ResponseReceived | ResponseReceived |
| `TimeoutError`, `ResponseError`, future variants | MaybeSent | MaybeSent |

Commit `1d3a17da` adds the one missing unit pin, a dispatch timeout on S3 Tables.

### The outcome table

Both catalogs share every row. "Glue / S3 Tables" names the catalog-specific code where they
differ. "Reconciled" means `Transaction::commit` takes its `CommitStateUnknown` arm
(`crates/iceberg/src/transaction/mod.rs:416`): it reloads the table and searches it for the
snapshot ids this attempt chose before sending (`latest_attempt_snapshot_ids`). Found gives `Ok`
with no re-apply; absent gives the original `CommitStateUnknown`. Drill pins live in each
catalog's `tests/commit_lost_reply_drill.rs`, named here without the `glue_` / `s3tables_`
prefix. They run the real SDK client against an in-process endpoint, and the shapes are measured
there. Unit pins live in Glue `src/error.rs` / `src/commit_transport.rs` and S3 Tables
`src/commit_transport.rs`.

**Ambiguous: may have been sent**

| Outcome | `SdkError` shape | Classifier → `ErrorKind` | Reconciled | Pin |
|---|---|---|---|---|
| Connection closed before the reply (FIN) | `DispatchFailure(ConnectorError { kind: Other(Some(TransientError)), source: hyper_util::client::legacy::Error(SendRequest, hyper::Error(IncompleteMessage)) })` (measured) | MaybeSent → `CommitStateUnknown` | yes | drill `lost_reply_on_a_closed_connection_reconciles_as_landed` (U-1; it also checks the found id against the request's), `unapplied_commit_on_a_closed_connection_is_unknown_and_keeps_the_pre_state` (U-4); unit Glue `test_closed_and_unclassified_dispatch_failures_classify_maybe_sent`, S3 Tables `test_commit_send_disposition_split` |
| Connection reset before the reply (RST) | `DispatchFailure(ConnectorError { kind: Io, source: … hyper::Error(Io, Os { kind: ConnectionReset }) })` (measured) | MaybeSent → `CommitStateUnknown` | yes | drill `lost_reply_is_not_resent_and_reconciles_as_landed` (U-1), `lost_unapplied_commit_is_unknown_and_keeps_the_pre_state` (U-4); unit Glue `test_refused_connection_classifies_never_sent_and_reset_stays_maybe_sent`, S3 Tables `test_commit_send_disposition_split` |
| Timeout, the operation | `SdkError::TimeoutError` (not drilled: the fork's clients set no operation timeout) | `_` arm: MaybeSent → `CommitStateUnknown` | yes | unit Glue `test_post_send_ambiguous_sdk_failures_classify_maybe_sent`, S3 Tables `test_commit_send_disposition_split`; transport (scripted `MaybeSentLost` = `SdkError::timeout_error`) `commit_outcome_tests::maybe_sent_is_unknown_without_a_second_commit_for_every_commit_class` |
| Timeout, the dispatch | `DispatchFailure(ConnectorError { kind: Timeout })` (not drilled: no read timeout is set, and a connect to 127.0.0.1 does not time out) | `_` arm: MaybeSent → `CommitStateUnknown` | yes | unit Glue `test_post_send_ambiguous_sdk_failures_classify_maybe_sent`, S3 Tables `test_commit_send_disposition_split` (assertion added in `1d3a17da`) |
| Server error, modeled | `ServiceError(InternalServiceException)` (Glue) / `ServiceError(InternalServerErrorException)` (S3 Tables), reply status 500 (measured) | MaybeSent (5xx arm) → `CommitStateUnknown`. Glue `OperationTimeoutException`: the mapper gives `CommitStateUnknown` for any status. | yes | drill `modeled_server_error_after_the_commit_reconciles_as_landed`; unit Glue `test_internal_service_and_operation_timeout_map_to_unknown_outcome`, S3 Tables `test_internal_server_error_maps_to_unknown_outcome_but_not_found_stays_terminal`; classifier Glue `test_server_error_reply_is_maybe_sent_and_client_error_reply_is_received`, S3 Tables `test_commit_send_disposition_split` |
| Server error, a code the SDK does not model | `ServiceError(Unhandled { code: Some("InternalFailure") })`, reply status 500 (measured) | **MaybeSent (5xx arm) → `CommitStateUnknown`** | **yes** | drill `server_error_with_an_unmodeled_code_after_the_commit_reconciles_as_landed`, `server_error_with_an_unmodeled_code_before_the_commit_is_unknown_and_keeps_the_pre_state`; unit as the row above |
| Unknown: dispatch `is_other()` with no kind | `DispatchFailure(ConnectorError { kind: Other(None), source: hyper_util::client::legacy::Error(SendRequest, hyper::Error(Parse(Status))) })`, an unparsable reply (measured) | **MaybeSent → `CommitStateUnknown`** | **yes** | drill `unparsable_reply_after_the_commit_reconciles_as_landed`, `unparsable_reply_before_the_commit_is_unknown_and_keeps_the_pre_state`; unit Glue `test_closed_and_unclassified_dispatch_failures_classify_maybe_sent`, S3 Tables `test_commit_send_disposition_split` |
| Unknown: a reply the SDK cannot read | `SdkError::ResponseError` (not drilled) | `_` arm: MaybeSent → `CommitStateUnknown` | yes | unit Glue `test_post_send_ambiguous_sdk_failures_classify_maybe_sent`, S3 Tables `test_commit_send_disposition_split` |
| Unknown: the `_` arm | `TimeoutError`, `ResponseError`, every dispatch failure but a user error or a refused connect, and any future variant of the `#[non_exhaustive]` `SdkError` | MaybeSent → `CommitStateUnknown` | yes | the rows above that reach `_`. A future variant cannot be built, so it has no pin of its own. |

**Definitive**

| Outcome | `SdkError` shape | Classifier → `ErrorKind` | Reconciled | Pin |
|---|---|---|---|---|
| CAS conflict | `ServiceError(ConcurrentModificationException)` 400 (Glue) / `ServiceError(ConflictException)` 409 (S3 Tables) (measured) | ResponseReceived → `CatalogCommitConflicts`, retryable | no. `Transaction::commit` retries; each retry refreshes and returns the refreshed table when it holds any id in `sent_ids` (every id an attempt chose before sending, `014136fe`); otherwise it re-bases | drill `landed_commit_answered_conflict_lands_once`, `an_earlier_attempt_that_lands_late_is_found_before_the_rebase` (two sent ids, the first lands late); core `transaction::action::rebase_landed_tests::*`; unit Glue `test_concurrent_modification_stays_retryable_conflict`, S3 Tables `test_conflict_exception_stays_retryable_conflict`; transport `cas_conflict_rebases_when_the_append_validation_contract_permits` |
| Connect refused | `DispatchFailure(ConnectorError { kind: Io, source: hyper_util::client::legacy::Error(Connect, ConnectError("tcp connect error", Os { kind: ConnectionRefused })) })` (measured) | NeverSent → `Unexpected` | no | drill `refused_commit_connection_is_never_sent_and_not_reconciled` (no commit request, no reconciliation read); unit Glue `test_refused_connection_classifies_never_sent_and_reset_stays_maybe_sent`, S3 Tables `test_commit_send_disposition_split` |
| Construction failure | `SdkError::ConstructionFailure` | NeverSent → `Unexpected` | no | unit Glue / S3 Tables `test_never_sent_sdk_error_maps_terminal_unexpected`, Glue `test_never_sent_and_service_response_do_not_classify_maybe_sent`; transport `never_sent_is_terminal_for_every_commit_class_on_the_shared_glue_path` / `…_s3tables_path` |
| Dispatch user error | `DispatchFailure(ConnectorError { kind: User })` | NeverSent → `Unexpected` | no | unit Glue `test_never_sent_and_service_response_do_not_classify_maybe_sent`, S3 Tables `test_commit_send_disposition_split` |
| 4xx refusal, modeled | Glue: `AccessDeniedException` → `Unexpected`, `EntityNotFoundException` → `TableNotFound`, `InvalidInputException` and the rest → `Unexpected`. S3 Tables: `ForbiddenException` → `Unexpected`, `NotFoundException` → `TableNotFound`, `BadRequestException` / `TooManyRequestsException` → `Unexpected` | ResponseReceived → the service-error mapping above | no | unit Glue `test_access_denied_is_terminal_not_unknown`, `test_definite_service_rejections_stay_terminal`; S3 Tables `test_forbidden_is_terminal_not_unknown`, `test_internal_server_error_maps_to_unknown_outcome_but_not_found_stays_terminal`; transport `permanent_authorization_denial_is_terminal_and_does_not_clean_staged_files` |
| 4xx refusal, a code the SDK does not model | `ServiceError(Unhandled { code: Some("ValidationException") })`, reply status 400 (measured) | ResponseReceived → `Unexpected` | no | drill `client_error_with_an_unmodeled_code_is_definitive_and_not_reconciled` (no reconciliation read) |

**Changed by `9830e241`** (before at `3aeb2928`, after at `9830e241`; both catalogs)

| Outcome | Before | After | Red at `3aeb2928` |
|---|---|---|---|
| 5xx reply, unmodeled code | ResponseReceived → mapper's last arm → `Unexpected` "Operation failed … for hitting aws sdk error"; not reconciled | MaybeSent → `CommitStateUnknown`; reconciled | `server_error_with_an_unmodeled_code_after_…`: `left: Err(Unexpected), right: Ok(())`; `…_before_…`: `left: Err(Unexpected), right: Err(CommitStateUnknown)` |
| Dispatch `is_other()` with no kind | NeverSent → `Unexpected` "… before the update request was sent"; not reconciled | MaybeSent → `CommitStateUnknown`; reconciled | `unparsable_reply_after_…`: `left: Err(Unexpected), right: Ok(())`; `…_before_…`: `left: Err(Unexpected), right: Err(CommitStateUnknown)` |
| 5xx reply, modeled | ResponseReceived → mapper → `CommitStateUnknown` | MaybeSent → `CommitStateUnknown` (same kind; the message is the classifier's "may have reached Glue / S3 Tables before the failure") | none (the kind is unchanged) |

### The id checked is the one chosen before sending

- **U-1 on a closed connection.** `*_lost_reply_on_a_closed_connection_reconciles_as_landed` reads
  the metadata file that the one commit request names (Glue `TableInput.Parameters.metadata_location`,
  S3 Tables `metadataLocation`; the drill logs it). It asserts that the snapshot the reconciliation
  found is that file's current snapshot: `marked_snapshot_ids == [sent]`. The fork writes that file
  before it sends the request, so the id was chosen before sending.
- **The re-base path after a CAS conflict.** `*_an_earlier_attempt_that_lands_late_is_found_before_the_rebase`:
  the first commit request is held and answered with a conflict. The second is answered with a
  conflict after the endpoint lands the first. Two ids were sent, and the one found is the first:
  `marked_snapshot_ids == [first]`, with no third commit request. A mutation that checks only the
  previous attempt's ids (`mem::take` of `latest_attempt_snapshot_ids`) turns it red on both
  catalogs: `left: [(Some("1"), DeferredThenConflict), (Some("1"), LandedDeferredThenConflict),
  (Some("2"), Answered)]` (S3 Tables: `v1`, `v1`, `v2`). The existing
  `landed_attempt_is_found_after_a_failed_refresh` covers only one sent id, so it did not prove
  "every id already sent".

### Gates (fork, on `9830e241`)

| Command | Exit |
|---|---|
| drills at the `3aeb2928` classifier (`cargo test --no-fail-fast -p iceberg-catalog-glue -p iceberg-catalog-s3tables --test commit_lost_reply_drill`) | 101: per catalog the four unmodeled-5xx and unparsable-reply pins red, the other 10 green |
| the same on `9830e241` | 0: 14 + 14 passed |
| `cargo test --no-fail-fast -p iceberg-catalog-glue -p iceberg-catalog-s3tables --lib` | 0 |
| the late-landing pins under the previous-attempt-only mutation | 101 (both red) |
| `cargo test --no-fail-fast -p iceberg -p iceberg-catalog-glue -p iceberg-catalog-s3tables` | 101: 5145 passed, 14 failed (the Docker-only `tests/glue_catalog_test.rs`), 21 ignored. Every test green at `3aeb2928` is green here except the one unit test this change renamed and flipped (`test_incomplete_message_classifies_maybe_sent_and_unclassified_other_stays_never_sent`, now `test_closed_and_unclassified_dispatch_failures_classify_maybe_sent`) |
| `cargo fmt --all -- --check` | 0 |
| `cargo clippy --all-targets --all-features --workspace -- -D warnings` | 0 |
| `make check-rust-file-size`, `make check-comment-blocks`, `make check-agent-artifacts`, `make check-matrix-anchors`; `typos` | 0 |
| comment gate, base `origin/main` (`076d5f98`) | 0 |

RePark never worse on `1d3a17da`, with the same command-line override as before (`cargo tree -i
iceberg` resolved to the fork path): `cargo test --no-fail-fast -p repark-iceberg -p repark-core
-p repark-spark -p repark-sql` exit 0, 5905 passed, 0 failed, 11 ignored, the same as the
`076d5f98` baseline, with no test green before and red after. P17
`append_midflight_commit_conflict_retries_and_both_appends_land` exit 0. RePark's `Cargo.lock`
was restored, and its tree is clean. On `1d3a17da` the fork also ran `cargo test -p
iceberg-catalog-s3tables --lib` (exit 0, 62 passed) and `cargo clippy -p iceberg-catalog-s3tables
--all-targets --all-features -- -D warnings` (exit 0).

## Owner ruling 127, 2026-10-16

The ruling, verbatim:

```
Q127 → (a). One Opus xhigh round on the same session, pins red on d36896aa
first, then the scoped re-verifier. Three precisions:

F2. A throttle reply (Glue code ThrottlingException on a 400, S3 Tables
    TooManyRequestsException 429) is a fourth disposition: refused before
    processing. The transport resends the SAME request — same metadata
    location, same version token or id, same snapshot ids — on the SDK
    schedule it replaces: three attempts, full-jitter exponential backoff
    from one second. Not a retryable error into the transaction loop:
    that path re-bases, re-writes metadata.json and reloads from the
    catalog that is throttling us. Exhausted → terminal, as base, and the
    message says nothing was applied. Drill cell on both catalogs:
    throttle, throttle, success → one commit, rows once, three sends.

F1. On BOTH catalogs. The SDK turns a 2xx with an unreadable body into a
    service error that still carries the 200. A service error whose raw
    status is 2xx classifies may-have-been-sent → CommitStateUnknown →
    read-back → landed. New drill cell on both catalogs: well-formed 200,
    JSON content type, body that is not JSON. The existing glue unit pin
    at 200 tests a shape the SDK does not produce here; add the service
    error shape beside it.

F3. The unknown-outcome read-back searches every id this transaction has
    sent, not the latest attempt's.

F4. The test moves so the doc comment re-attaches to its item.

Ledger: the outcome table gains the two rows (throttle; 2xx unreadable)
with the measured SdkError shape, before and after.

Re-verifier: scoped to F1–F4 plus the classifier mutants. PASS → the fork
push and rev bump come to me as a question. Not green → park; no second
round without Fable's look.
```

Commits: `66e85940` (the pins, red on `d36896aa`), `9a9f91baf920` (the fix), `64a37273` (F4).

### Red first, on `d36896aa`

| Pin (per catalog unless named) | Red on `d36896aa` |
|---|---|
| `*_throttled_commit_is_resent_unchanged_and_lands_once` | `left: [(Some("1"), DroppedThenError(400))]`, `right:` three sends (S3 Tables: `v1`, 429) |
| `*_throttled_on_every_send_fails_terminally_after_three_sends` | `left: [(Some("1"), DroppedThenError(400))]`, `right:` three sends (S3 Tables: `v1`, 429) |
| `*_a_late_landing_attempt_is_found_by_the_unknown_outcome_read_back` | `left: Err(CommitStateUnknown), right: Ok(())` |
| `s3tables_unreadable_success_reply_after_the_commit_reconciles_as_landed` | `left: 2, right: 3` (reads: no read-back; the outcome was `Err(Unexpected)`) |
| `s3tables_unreadable_success_reply_before_the_commit_is_unknown_and_keeps_the_pre_state` | `left: 2, right: 3` (the outcome was `Err(Unexpected)`) |
| Glue unit `test_post_send_ambiguous_sdk_failures_classify_maybe_sent` (new case `service-error-2xx`) | `service-error-2xx: a post-send failure must classify as MaybeSent (unknown outcome)` |
| S3 Tables unit `test_commit_send_disposition_split` (new 200 assertion) | `assertion failed: matches!(classify_commit_send_disposition(&TestSdkError::service_error((), response(200))), CommitSendDisposition::MaybeSent)` |
| `glue_unreadable_success_reply_after_the_commit_is_success_without_a_read_back` | green: Ok, one snapshot, 3 rows, 2 reads. The Glue SDK reads an empty `UpdateTable` output and reports success, so no `SdkError` forms. |

Measured with an uncommitted probe on `d36896aa`: on Glue, the same unreadable 200 **without
applying** also returns Ok, with no read-back and nothing on the table. A 200 is the service's
success; the classifier never sees an error. This is the hand-back's question Q1.

### The change (commit `9a9f91baf920`)

- **F2, refused before processing.** Each live commit transport (Glue
  `src/commit_transport.rs` `LiveGlueCommitTransport::send_update_table`, S3 Tables
  `LiveS3TablesCommitTransport::send_update_metadata_location`) sends through `send_once`. It
  builds the request from the same call every time, so the metadata location, the version id or
  token, and the snapshot ids are the same. `refused_before_processing` matches a 4xx
  `ServiceError` with Glue code `ThrottlingException` or S3 Tables `TooManyRequestsException`
  only. On that reply the transport sleeps `throttle_backoff(sends, jitter_fraction())`, a
  uniform draw below 1 s after the first send and below 2 s after the second, on the client's
  own `sleep_impl`. Then it sends again, up to `THROTTLE_ATTEMPTS = 3` sends. The SDK retry stays
  off. After the third throttle, `RefusedBeforeProcessing` maps to a terminal `Unexpected`
  ("… throttled the update on each of 3 sends, and nothing was applied"), as on base. Nothing
  goes back to the transaction loop, so nothing re-bases, rewrites `metadata.json` or reloads.
- **F1.** Both classifiers: a `ServiceError` whose raw status is 2xx is MaybeSent, as a 5xx is.
  `map_*_commit_sdk_error` gives `CommitStateUnknown`.
- **F3.** `Transaction::reconcile_unknown_commit_outcome` searches `sent_ids`. The field doc of
  `latest_attempt_snapshot_ids`, which described it as the reconciliation evidence, is deleted.
  `mod.rs` loses those 8 lines, and its file-size ceiling drops to 1928.
- **F4 (commit `64a37273`).** The three tests between the "Risk (… the over-broadening
  direction)" doc comment and `test_never_sent_and_service_response_do_not_classify_maybe_sent`
  move below that test. The comment keeps its text and its lines.

### The outcome table, two new rows

| Outcome | `SdkError` shape (measured) | `d36896aa` | `9a9f91baf920` | Pins |
|---|---|---|---|---|
| Throttle | Glue: `ServiceError(Unhandled { code: Some("ThrottlingException") })`, status 400. S3 Tables: `ServiceError(TooManyRequestsException)`, status 429 | ResponseReceived → `Unexpected` after one send; not reconciled, not resent (base `076d5f98`: the SDK resent and committed) | **Refused before processing**: the transport resends the same request, three sends, full-jitter backoff from 1 s. Landed → Ok, rows once. Exhausted → terminal `Unexpected`, "nothing was applied", no read-back | drill `*_throttled_commit_is_resent_unchanged_and_lands_once`, `*_throttled_on_every_send_fails_terminally_after_three_sends`; unit `test_throttled_commit_schedule_is_three_sends_with_full_jitter_from_one_second`, `test_only_a_throttle_reply_is_refused_before_processing`, `test_an_exhausted_throttle_is_terminal_and_says_nothing_was_applied` |
| 2xx reply with an unreadable body | S3 Tables: `ServiceError(Unhandled { source: DeserializeError { kind: UnexpectedToken('n', …) }, code: None })`, status 200. Glue: no `SdkError`; the SDK reports success | S3 Tables: ResponseReceived → `Unexpected`; not reconciled. Glue: Ok | S3 Tables: **MaybeSent → `CommitStateUnknown`; reconciled** (landed → Ok; absent → `CommitStateUnknown`). Glue: Ok, unchanged (no error forms) | drill `s3tables_unreadable_success_reply_after_the_commit_reconciles_as_landed`, `s3tables_unreadable_success_reply_before_the_commit_is_unknown_and_keeps_the_pre_state`, `glue_unreadable_success_reply_after_the_commit_is_success_without_a_read_back`; unit Glue `test_post_send_ambiguous_sdk_failures_classify_maybe_sent` (`service-error-2xx`), S3 Tables `test_commit_send_disposition_split` |

In the ruling-115 table above, the reconciliation note names `latest_attempt_snapshot_ids`.
Since `9a9f91baf920` the read-back searches `sent_ids` (F3). The clerk's card copy carries the
updated note.

### Mutants (each applied, run, restored; the tree was clean after each)

| Mutant | Killed by |
|---|---|
| Throttle as a retryable conflict into the transaction loop (Glue, S3 Tables) | `*_throttled_commit_is_resent_unchanged_and_lands_once`, `*_throttled_on_every_send_fails_terminally_after_three_sends`, `test_an_exhausted_throttle_is_terminal_and_says_nothing_was_applied` |
| The resend rebuilds the request with a new metadata location (Glue, S3 Tables) | `*_throttled_commit_is_resent_unchanged_and_lands_once`, `*_throttled_on_every_send_fails_terminally_after_three_sends` |
| Four sends (Glue, S3 Tables) | `test_throttled_commit_schedule_…`, `*_throttled_on_every_send_fails_terminally_after_three_sends` |
| Two sends (Glue, S3 Tables) | `test_throttled_commit_schedule_…`, both throttle drills |
| A 2xx `ServiceError` back to ResponseReceived, Glue | `test_post_send_ambiguous_sdk_failures_classify_maybe_sent` |
| The same, S3 Tables | `test_commit_send_disposition_split`, both `s3tables_unreadable_success_reply_*` drills |
| The read-back on `latest_attempt_snapshot_ids` | `glue_` / `s3tables_a_late_landing_attempt_is_found_by_the_unknown_outcome_read_back` |
| Verifier M1 (5xx → ResponseReceived), Glue / S3 Tables | the 5xx classifier unit pins and both unmodeled-5xx drills |
| Verifier M2 (`other(None)` → NeverSent), Glue / S3 Tables | the closed-and-unclassified unit pins and both unparsable-reply drills |
| Verifier M7 (refused connect → MaybeSent), Glue / S3 Tables | the refused-connection unit pins and `*_refused_commit_connection_is_never_sent_and_not_reconciled` |
| Verifier M8 (4xx → MaybeSent), Glue / S3 Tables | the never-sent / response-received unit pins, `*_client_error_with_an_unmodeled_code_is_definitive_and_not_reconciled`, both late-landing drills |

19 mutants, 19 killed.

### Gates (fork, on `64a37273`)

| Command | Exit |
|---|---|
| `cargo test --no-fail-fast -p iceberg -p iceberg-catalog-glue -p iceberg-catalog-s3tables` | 101: 5160 passed, 14 failed (the Docker-only `tests/glue_catalog_test.rs`), 21 ignored. Every test green on `d36896aa` is green here (the drill cells moved from `tests/commit_lost_reply_drill.rs` to `tests/commit_lost_reply_drill/main.rs`; same target, same names). |
| `cargo fmt --all -- --check` | 0 |
| `cargo clippy --all-targets --all-features --workspace -- -D warnings` | 0 |
| `make check-rust-file-size check-comment-blocks check-agent-artifacts check-matrix-anchors` | 0 |
| comment gate, base `076d5f98` | 0 |

RePark never worse, on `64a37273`. The earlier rounds' RePark clone was gone, so this ran in a
fresh clone of RePark main `ae880bf6`, which pins the fork at `076d5f98`, with the same
command-line override as before. `cargo tree -i iceberg` resolved to the fork path. `cargo test
--no-fail-fast -p repark-iceberg -p repark-core -p repark-spark -p repark-sql`:

- without the override: exit 101, 5913 passed, 1 failed, 11 ignored. The failure is
  `repark-core` `microbatch::lifecycle_tests::two_sessions_on_one_sink_wait_for_the_scope`, the
  known load-sensitive timing flake.
- with it: exit 0, 5914 passed, 0 failed, 11 ignored.

No test is green without the override and red with it. P17
`append_midflight_commit_conflict_retries_and_both_appends_land` exit 0. RePark's `Cargo.lock`
was restored, and the clone is clean.
