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


# F-MINIO-QUAY — the MinIO fixtures pull from quay.io

**Date:** 2026-09-11. **Branch:** `fix/f-minio-quay`. **Base:** `origin/main` `6232e39f`.
**Model:** orchestrator (Claude Opus 5), no worker round. **Path:** FIXTURE, the same class as F-HMS.

## Defect

Docker Hub no longer serves the MinIO repositories. `https://hub.docker.com/v2/repositories/minio/minio/`
and `…/minio/mc/` return 404 (measured 2026-09-11), and `make docker-up` fails with
`pull access denied for minio/mc, repository does not exist`. Every `Tests (default)` run fails before any
integration test starts. The first run to hit it was PR #276 (CI run 34645915527).

## Fix

`dev/docker-compose.yaml` pulls the same two release tags from `quay.io/minio`, pinned by digest:

| Service | Image |
|---|---|
| `minio` | `quay.io/minio/minio:RELEASE.2025-05-24T17-08-30Z@sha256:a616cd8f37758b0296db62cc9e6af05a074e844cc7b5c0a0e62176d73828d440` |
| `mc` | `quay.io/minio/mc:RELEASE.2025-05-21T01-59-54Z@sha256:09f93f534cde415d192bb6084dd0e0ddd1715fb602f8a922ad121fd2bf0f8b44` |

The tags and entrypoints do not change, and neither do the healthcheck or the bucket script.

## Proposition ledger

| Clause | Checkable proposition | Proof obligation | Verdict | Evidence |
|---|---|---|---|---|
| C-001 | Both pinned digests are what quay.io serves for the tags. | Fetch each manifest index anonymously from `https://quay.io/v2/minio/<repo>/manifests/<digest>` and hash it. | PROVEN | HTTP 200 for both. `sha256sum` of each body equals its pinned digest (`a616cd8f…`, `09f93f53…`). The quay tag API lists both tags as active, and each is a 3-platform manifest list. |
| C-002 | The fixture comes up and the integration suite runs on it. | This PR's `Tests (default)` CI run. There is no Docker on the authoring host. | OPEN until CI | The PR's CI run. |

Every other fixture image (`apache/iceberg-rest-fixture`, `motoserver/moto`, `fsouza/fake-gcs-server`,
`apache/hive`) still answers 200 on Docker Hub (measured 2026-09-11).
