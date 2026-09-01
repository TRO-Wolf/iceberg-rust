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

# map.md — crates/catalog/s3tables/

## Purpose

S3 Tables catalog implementation. PR-5A owns the commit-transport seam on `UpdateTableMetadataLocation`.

## Contents

| File | What it does |
|---|---|
| `src/catalog.rs` | S3 Tables `Catalog` impl. Commit CAS goes through `S3TablesCommitTransport`. |
| `src/commit_transport.rs` | Narrow seam around the completed `UpdateTableMetadataLocation` SDK call. Classifier, mapping, live / discarding / scripted transports. |
| `src/commit_outcome_tests.rs` | Offline outcome proofs for the seven commit classes on this one path. |
| `src/utils.rs` | SDK config. |

Java iceberg-aws 1.10.0 has no `S3TablesTableOperations`. Java talks to S3 Tables through the REST catalog. REST `CommitErrorHandler` maps 409 to `CommitFailedException` and 5xx to `CommitStateUnknownException`. `DefaultErrorHandler` maps 403 to `ForbiddenException`.

S3 Tables service-side maintenance is a concurrent committer even under one application writer (`docs/ENGINE_CONTRACT.md` §8).

## I want to…

| Intent | Go to |
|---|---|
| Inject a never-sent / lost-response / conflict / forbidden commit | `src/commit_transport.rs` `S3TablesCommitScript` |
| Classify an S3 Tables SDK commit failure | `classify_commit_send_disposition` then `map_s3tables_commit_sdk_error` |
| Run credentialed smokes | `dev/pr5a-catalog-commit-outcomes.sh` |

## Pointers

- Up: `crates/catalog/`
- Related: `crates/catalog/glue/map.md`, `docs/parity/GAP_MATRIX.md` row R110 and row R157, `task/pr5a-catalog-commit-outcomes-ledger.md`

## Debug

### Known failure modes

| Symptom | Likely cause | First check |
|---|---|---|
| Duplicate commit after response loss | Retry of `CommitStateUnknown` | `catalog_commit_attempts == 1` |
| Conflict treated as terminal | `ConflictException` lost `retryable` | `map_update_table_metadata_location_service_error` |
| Forbidden retried | Auth mapped unknown | Forbidden arm stays `Unexpected` |

### First checks

- Offline: `cargo test -p iceberg-catalog-s3tables --lib --locked`
- Decode: `dev/java-interop/run-pr5a-catalog-commit-decode.sh`

### Escalate to

`docs/ENGINE_CONTRACT.md` §8, `task/pr5a-catalog-commit-outcomes-ledger.md`
