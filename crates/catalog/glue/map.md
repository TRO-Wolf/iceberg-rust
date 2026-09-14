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
| `src/commit_transport.rs` | Narrow seam around the completed Glue `UpdateTable` SDK call. Live / discarding / scripted transports. Classifier feed + service-error mapping. The scripted transport records the last `GlueUpdateTableCall` (`last_call()` → version id + `TableInput` parameters) so pins can see the CAS token and `metadata_location`/`previous_metadata_location` wires. |
| `src/commit_outcome_tests.rs` | Offline outcome proofs for the seven commit classes on this one path. Credentialed tests arm on `ICEBERG_PR5A_CREDENTIALED`. `catalog_with_version` seeds the harness pointer at any version id. |
| `src/replace_publish_tests.rs` | Offline outcome pins for staged replace publish on the scripted transport: pointer swap + uuid/log retention + the sent `version_id`/`metadata_location`/`previous_metadata_location` wires (mutation-pinned against M1/M6), stale-base conflict before send, unreadable/foreign-uuid staged file refused before send, lost response typed `CommitStateUnknown` (no reconciliation), `ConcurrentModification` retryable, `AccessDenied` terminal. |
| `src/error.rs` | `classify_commit_send_disposition` (NeverSent / MaybeSent / ResponseReceived). |
| `src/schema.rs` | Iceberg schema to Glue columns. |
| `src/utils.rs` | SDK config, `convert_to_glue_table`, namespace validation. |

## I want to…

| Intent | Go to |
|---|---|
| Inject a never-sent / lost-response / modeled service commit | `src/commit_transport.rs` `GlueCommitScript` + `GlueCatalog::for_commit_outcome_tests_at_version` |
| Classify a Glue SDK commit failure | `src/error.rs` `classify_commit_send_disposition` then `map_glue_commit_sdk_error` |
| Run credentialed smokes | `dev/pr5a-catalog-commit-outcomes.sh` |

## Pointers

- Up: `crates/catalog/`
- Related: `crates/catalog/s3tables/map.md`, `docs/parity/GAP_MATRIX.md` row R110 and row R157, `task/pr5a-catalog-commit-outcomes-ledger.md`

## Debug

### Known failure modes

| Symptom | Likely cause | First check |
|---|---|---|
| Duplicate rows after a timeout | Unknown outcome retried | `catalog_commit_attempts` must stay 1 on `CommitStateUnknown` |
| Never-sent classified unknown | Classifier drift | `CommitSendDisposition::NeverSent` in `error.rs` |
| Auth denial retried | AccessDenied mapped retryable | `map_update_table_service_error` AccessDenied arm |

### First checks

- Offline: `cargo test -p iceberg-catalog-glue --lib --locked`
- Decode: `dev/java-interop/run-interop-pr5a-catalog-commit-decode.sh`

### Escalate to

`docs/ENGINE_CONTRACT.md` §8, `task/pr5a-catalog-commit-outcomes-ledger.md`
