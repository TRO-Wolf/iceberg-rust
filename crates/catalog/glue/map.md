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
| `src/catalog.rs` | Glue `Catalog` impl. `update_table` writes metadata then sends through `GlueCommitTransport`. |
| `src/commit_transport.rs` | Narrow seam around the completed Glue `UpdateTable` SDK call. Live / discarding / scripted transports. Classifier feed + service-error mapping. |
| `src/commit_outcome_tests.rs` | Offline outcome proofs for the seven commit classes on this one path. Credentialed tests arm on `ICEBERG_PR5A_CREDENTIALED`. |
| `src/error.rs` | `classify_commit_send_disposition` (NeverSent / MaybeSent / ResponseReceived). |
| `src/schema.rs` | Iceberg schema to Glue columns. |
| `src/utils.rs` | SDK config, `convert_to_glue_table`, namespace validation. |

## I want to…

| Intent | Go to |
|---|---|
| Inject a never-sent / lost-response / modeled service commit | `src/commit_transport.rs` `GlueCommitScript` + `GlueCatalog::for_commit_outcome_tests` |
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
- Decode: `dev/java-interop/run-pr5a-catalog-commit-decode.sh`

### Escalate to

`docs/ENGINE_CONTRACT.md` §8, `task/pr5a-catalog-commit-outcomes-ledger.md`
