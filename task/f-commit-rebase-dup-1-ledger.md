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
