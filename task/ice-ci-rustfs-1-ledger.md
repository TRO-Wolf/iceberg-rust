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

# ICE-CI-RUSTFS-1 — RustFS replaces MinIO as the shared S3 test service

**Date:** 2026-10-17. **Branch:** `ci/rustfs-object-store`. **Base:** `origin/main` `076d5f982`.
**Model:** Claude Opus 5.5 executor. **Path:** FIXTURE, the same class as
[F-MINIO-QUAY](f-minio-quay-ledger.md), which this unit supersedes.

## Defect

The `Tests (default)` CI job fails at `make docker-up` on every run since 2026-09-24. quay.io no
longer serves `quay.io/minio/minio:RELEASE.2025-05-24T17-08-30Z` or
`quay.io/minio/mc:RELEASE.2025-05-21T01-59-54Z` ("no such manifest"). No nextest or doc test ran on
CI after that date, so PRs #359–#366 merged with a red test job.

Upstream hit the same break on the same day. It moved to RustFS in `apache/iceberg-rust` PR #3273,
commit `119fad92c`. A cherry-pick conflicts on this fork, so this unit ports it by hand.

## Fix

| File | Change |
|---|---|
| `dev/docker-compose.yaml` | Service `minio` → `object-store` on `rustfs/rustfs:1.0.0`, digest-pinned. Same `admin`/`password` credentials, ports 9000/9001. `RUSTFS_SERVER_DOMAINS=object-store` and the aliases `icebergdata`/`warehouse`/`bucket1.object-store` keep virtual-hosted access. The healthcheck reads `/health/ready`. Service `mc` → one-shot `create-buckets`, which runs `curl --aws-sigv4` from the RustFS image. `spark-iceberg` waits for `create-buckets` to exit 0. The REST fixture, HMS and Spark point at `http://object-store:9000`. |
| `dev/hms/core-site.xml`, `dev/spark/spark-defaults.conf` | `fs.s3a.endpoint` and the REST catalog `s3.endpoint` → `http://object-store:9000`. |
| `Makefile` | `docker-up` adds `--remove-orphans`, so an old `minio` container cannot hold port 9000. |
| `crates/test_utils/src/lib.rs` | `ENV_MINIO_ENDPOINT` → `ENV_OBJECT_STORE_ENDPOINT` (env var `ICEBERG_TEST_OBJECT_STORE_ENDPOINT`), `DEFAULT_MINIO_PORT` → `DEFAULT_OBJECT_STORE_PORT`, `get_minio_endpoint` → `get_object_store_endpoint`. These are upstream's names. |
| Callers | glue, hms, `integration_tests` and opendal `file_io_s3_test` (incl. `MockCredentialLoader::new_object_store`). The glue test and the fork's two bucket-root pins are fork-only callers. Upstream's `catalog/loader/tests/common/mod.rs` and `opendal/tests/resolving_storage_test.rs` do not exist on this fork. |
| Markdown | `AGENTS.md`, `docs/testing.md`, `dev/map.md`, `website/src/reference/container-runtimes.md`. |

The compose file drops the MinIO banner and the `mc` comment line. It adds no comment. Upstream's
`create-buckets` comment is not carried over (comment ban).

The anonymous-read bucket policy (`mc policy set public`) is gone, as upstream. No test reads
anonymously. `test_s3_with_custom_credential_loader_integration_failure` (in
`file_io_s3_test.rs`) asserts that a credential-less read is refused, and it passes.

Breaking test-surface change: an environment that set `ICEBERG_TEST_MINIO_ENDPOINT` must set
`ICEBERG_TEST_OBJECT_STORE_ENDPOINT`. No workflow in `.github/workflows/` sets either variable.

### Image pin

| Service | Image |
|---|---|
| `object-store`, `create-buckets` | `rustfs/rustfs:1.0.0@sha256:8cc9801755448b71a786705ce76692c77e14936cccd87cf2fc31842e58f4d1ff` |

The digest is the OCI image index that `docker buildx imagetools inspect rustfs/rustfs:1.0.0`
reports (linux/amd64 and linux/arm64), measured 2026-10-17.

## Proposition ledger

| Clause | Checkable proposition | Proof obligation | Verdict | Evidence |
|---|---|---|---|---|
| C-001 | The stack comes up on RustFS. | `make docker-up`, then `docker compose ps -a`. | PROVEN | `object-store` healthy, `create-buckets` `Exited (0)`, rest / hive-metastore / moto / spark-iceberg / provision healthy (2026-10-17, rootless Docker). |
| C-002 | Every suite that reads the object-store endpoint passes on it. | `cargo nextest run --all-features -p <pkg>` with the stack up. | PROVEN | opendal 67/67 (8 in `file_io_s3_test`), integration_tests 3/3, hms 60/60, glue 83/83. |
| C-003 | Spark provisions through the REST fixture onto RustFS. | `provision` turns healthy; the integration suite reads the tables it wrote. | PROVEN | `provision` healthy; `read_positional_deletes` and `read_evolved_schema` pass. |
| C-004 | No code file gains a comment. | `make check-comment-blocks`; diff review. | PROVEN | Gate green. The one new line is the compiler-demanded doc line on `get_object_store_endpoint`. |
| C-005 | CI `Tests (default)` is green. | This PR's CI run. | OPEN until CI | The PR's CI run. |
