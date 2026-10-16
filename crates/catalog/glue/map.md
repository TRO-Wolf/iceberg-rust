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

# map.md — crates/catalog/glue/

## Purpose

AWS Glue catalog implementation. PR-5A owns the commit-transport seam on `update_table`.

## Contents

| File | What it does |
|---|---|
| `src/catalog.rs` | Glue `Catalog` impl. `update_table` writes metadata then sends through `GlueCommitTransport`; `publish_replace_table` delegates to `catalog/replace_publish.rs`. |
| `src/catalog/replace_publish.rs` | Staged replace publish: `get_table_pointer` → expected-base conflict (retryable, before any send) → staged-metadata read-validate (`read_from` + uuid match, `DataInvalid` before send) → `UpdateTable` through the commit transport with the stored version-id (the CAS). |
| `src/catalog/tests.rs` | Unit tests extracted from `catalog.rs` (catalog ctor, config/`Debug` redaction, namespace-not-empty). |
| `src/catalog/test_support.rs` | `#[cfg(test)]` catalog seams extracted from `catalog.rs`: `catalog_commit_attempts`, `with_commit_transport` / `live_commit_transport`, `for_commit_outcome_tests_at_version` (injectable pointer version id, `Some("v0")` via `catalog_with` — a `None` GetTable version means the send carries `version_id: None`, no Glue OCC). |
| `src/commit_transport.rs` | Narrow seam around the completed Glue `UpdateTable` SDK call. Live / discarding / scripted transports. The live transport sends `UpdateTable` with the SDK retry off for that one operation (`customize().config_override(... RetryConfig::disabled())`), so a lost reply reaches the fork as an unknown outcome, never as an SDK re-send; reads keep the client's retry (ICE-COMMIT-REBASE-DUP-1). Classifier feed + service-error mapping. The scripted transport records the last `GlueUpdateTableCall` (`last_call()` → version id + `TableInput` parameters) so pins can see the CAS token and `metadata_location`/`previous_metadata_location` wires. On a throttle reply, `refused_before_processing` is the fourth disposition: the reply's error code is on the SDK's own `aws_runtime::retries::classifiers::THROTTLING_ERRORS`, on any status, checked before the classifier's 5xx rule (owner rulings 127, 129 and 130). `RequestTimeout` and `RequestTimeoutException` are not on that list and are not resent; a 5xx with no code or another code stays MaybeSent. The live transport resends the same request through `send_once`, for up to `THROTTLE_ATTEMPTS` = 3 sends. Between sends it sleeps `throttle_backoff`, full jitter below 1 s and then below 2 s, on the client's `sleep_impl`. `RefusedBeforeProcessing` then maps to a terminal `Unexpected` that says nothing was applied. A throttle never returns to the transaction loop. |
| `src/commit_outcome_tests.rs` | Offline outcome proofs for the seven commit classes on this one path. Credentialed tests arm on `ICEBERG_PR5A_CREDENTIALED`. `catalog_with_version` seeds the harness pointer at any version id. |
| `src/replace_publish_tests.rs` | Offline outcome pins for staged replace publish on the scripted transport: pointer swap + uuid/log retention + the sent `version_id`/`metadata_location`/`previous_metadata_location` wires (mutation-pinned against M1/M6), stale-base conflict before send, unreadable/foreign-uuid staged file refused before send, lost response typed `CommitStateUnknown` (no reconciliation), `ConcurrentModification` retryable, `AccessDenied` terminal. |
| `tests/commit_lost_reply_drill/` (`main.rs` the cells, `endpoint.rs` the harness) | The lost-reply drill (ICE-COMMIT-REBASE-DUP-1): an in-process Glue endpoint with a `VersionId` CAS, driven by the real SDK client the catalog builds with dummy credentials. It applies or drops a commit and then resets the connection (RST), closes it (FIN), answers an error status with a given code, or writes an unparsable reply, or answers 200 `application/json` with a body that is not JSON; it can apply a commit and answer `ConcurrentModificationException`, hold a commit and land it late while rejecting or resetting the next, or close its listener so the commit connection is refused. It logs every request and the metadata file each commit request names. |
| `src/error.rs` | `classify_commit_send_disposition` (NeverSent / MaybeSent / ResponseReceived) over `SdkError<E, HttpResponse>`. A dispatch failure is NeverSent only for `is_user()` or an `io::Error` of kind `ConnectionRefused` in its source chain; every other dispatch failure (reset, timeout, a closed connection `other(Some(TransientError))`, an unparsable reply `other(None)`) is MaybeSent. A `ServiceError` whose reply status is 2xx or 5xx is MaybeSent, modeled or not (2xx: owner ruling 127; the Glue SDK reports a 2xx with an unreadable body as success, so no error forms); a 4xx reply stays ResponseReceived and keeps the service-error mapping (ICE-COMMIT-REBASE-DUP-1, owner ruling 115). |
| `src/schema.rs` | Iceberg schema to Glue columns. |
| `src/utils.rs` | SDK config, `convert_to_glue_table`, namespace validation. |

## I want to…

| Intent | Go to |
|---|---|
| Inject a never-sent / lost-response / modeled service commit | `src/commit_transport.rs` `GlueCommitScript` + `GlueCatalog::for_commit_outcome_tests_at_version` |
| Classify a Glue SDK commit failure | `src/error.rs` `classify_commit_send_disposition` then `map_glue_commit_sdk_error` |
| Run credentialed smokes | `dev/pr5a-catalog-commit-outcomes.sh` |
| Drill a lost commit reply through the real SDK, offline | `tests/commit_lost_reply_drill/` |

## Pointers

- Up: `crates/catalog/`
- Related: `crates/catalog/s3tables/map.md`, `docs/parity/GAP_MATRIX.md` row R110 and row R157, `task/pr5a-catalog-commit-outcomes-ledger.md`

## Debug

### Known failure modes

| Symptom | Likely cause | First check |
|---|---|---|
| Duplicate rows after a timeout | Unknown outcome retried | `catalog_commit_attempts` must stay 1 on `CommitStateUnknown` |
| Duplicate rows after a lost reply | The SDK re-sent `UpdateTable` and the re-base re-applied the append | `tests/commit_lost_reply_drill/`: one `UpdateTable` per lost reply |
| Never-sent classified unknown | Classifier drift | `CommitSendDisposition::NeverSent` in `error.rs` |
| A landed commit fails as "before the update request was sent" or "for hitting aws sdk error" | A closed connection, an unparsable reply or a 5xx reply classified as definitive | `tests/commit_lost_reply_drill/` closed-connection pins; the `TransientError` arm in `error.rs` |
| Auth denial retried | AccessDenied mapped retryable | `map_update_table_service_error` AccessDenied arm |

### First checks

- Offline: `cargo test -p iceberg-catalog-glue --lib --locked`
- Decode: `dev/java-interop/run-interop-pr5a-catalog-commit-decode.sh`

### Escalate to

`docs/ENGINE_CONTRACT.md` §8, `task/pr5a-catalog-commit-outcomes-ledger.md`
